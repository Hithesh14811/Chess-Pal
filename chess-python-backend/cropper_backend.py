import cv2
import numpy as np
import os
import shutil
import mss
import mss.tools
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
from pathlib import Path
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Folder where piece images will be stored
PIECES_FOLDER = os.path.join(BASE_DIR, "identify")
# Ensure the directory exists
Path(PIECES_FOLDER).mkdir(parents=True, exist_ok=True)

# Load the machine learning model with optimizations
MODEL_CANDIDATES = [
    os.path.join(BASE_DIR, "chess_model_efficientnet_best2.keras"),
    # Output of train_efficientnet.py
    os.path.join(BASE_DIR, "chess_model_efficientnet_trained.keras"),
]
MODEL_PATH = next((p for p in MODEL_CANDIDATES if os.path.exists(p)), None)
if MODEL_PATH is None:
    raise FileNotFoundError(
        "No chess model found. Expected one of: " + ", ".join(MODEL_CANDIDATES)
    )

# Configure TensorFlow for better CPU performance
tf.config.threading.set_intra_op_parallelism_threads(2)
tf.config.threading.set_inter_op_parallelism_threads(2)

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    # Read the model's expected input size instead of hardcoding it
    model_input_shape = model.input_shape
    if isinstance(model_input_shape, (tuple, list)):
        model_input_shape = tuple(int(d) for d in model_input_shape[1:3])
    # Create a prediction function that processes multiple images at once
    predict_batch = lambda imgs: model.predict(imgs, batch_size=64, verbose=0)
    logger.info(f"Successfully loaded model from {MODEL_PATH}")
    logger.info(f"Model expects input size: {model_input_shape}")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    model = None
    predict_batch = None
    model_input_shape = (128, 128)  # Fallback default

# Map from model prediction indices to piece names
PIECE_MAPPING = {
    0: "bB", 1: "bK", 2: "bN", 3: "bP", 4: "bQ", 5: "bR",
    6: "empty", 
    7: "wB", 8: "wK", 9: "wN", 10: "wP", 11: "wQ", 12: "wR"
}

def order_points(pts):
    """Order points: top-left, top-right, bottom-right, bottom-left"""
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]  # top-left
    rect[2] = pts[np.argmax(s)]  # bottom-right
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]  # top-right
    rect[3] = pts[np.argmax(diff)]  # bottom-left
    return rect

def four_point_transform(image, pts):
    """Apply perspective transformation to get top-down view"""
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    
    # Compute width and height of new image
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = int(max(widthA, widthB))
    
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = int(max(heightA, heightB))
    
    # Destination points for warped image
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    
    # Perspective transform
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    return warped

def detect_chess_board(img):
    """Accurate chess board detection using code 1 logic"""
    logger.info("Detecting chess board in captured image...")
    
    # Convert to numpy array if not already
    if not isinstance(img, np.ndarray):
        img = np.array(img)
    
    # Keep original image for edge drawing
    original_img = img.copy()
    
    # Get image dimensions
    height, width = img.shape[:2]
    image_area = width * height
    
    # Preprocess with adaptive thresholding
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 11, 3
    )
    
    # Find contours with hierarchy
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    chessboard_contour = None
    
    for cnt in sorted(contours, key=cv2.contourArea, reverse=True):
        area = cv2.contourArea(cnt)
        if area < image_area * 0.002 or area > image_area * 0.9:
            continue
        
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        
        if len(approx) == 4:
            (x, y, w, h) = cv2.boundingRect(approx)
            aspect_ratio = float(w) / h
            if 0.6 < aspect_ratio < 1.4:
                chessboard_contour = approx
                break
    
    if chessboard_contour is not None:
        # Draw contours on original image for 1b
        edges_img = original_img.copy()
        cv2.drawContours(edges_img, [chessboard_contour], -1, (0, 255, 0), 3)
        
        # Save image with edges drawn (1b)
        edges_path = os.path.join(PIECES_FOLDER, "1b.png")
        cv2.imwrite(edges_path, edges_img)
        logger.info("✅ Chessboard detected and outlined - saved as 1b.png")
        
        # Apply perspective transformation to get top-down view
        warped = four_point_transform(original_img, chessboard_contour.reshape(4, 2))
        
        # Save the warped chessboard (1c)
        cropped_path = os.path.join(PIECES_FOLDER, "1c.png")
        cv2.imwrite(cropped_path, warped)
        logger.info("📸 Cropped chessboard saved as 1c.png")
        
        return warped
    else:
        logger.warning("⚠️ Chessboard not detected.")
        return None

def clean_piece_folder():
    """Clean the pieces folder by removing any existing files"""
    if os.path.exists(PIECES_FOLDER):
        for file in os.listdir(PIECES_FOLDER):
            file_path = os.path.join(PIECES_FOLDER, file)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                logger.error(f"Error deleting {file_path}: {e}")
    logger.info("Cleaned pieces folder")

def identify_pieces():
    """Optimized piece identification using batch processing"""
    if model is None:
        logger.error("No ML model loaded. Cannot identify pieces.")
        return {}
    
    pieces = {}
    img_size = model_input_shape  # Model's expected input size (read from model)
    batch_size = 64
    image_paths = []
    valid_indices = []
    
    # First pass: collect all valid images
    for i in range(1, 65):
        img_path = os.path.join(PIECES_FOLDER, f"{i}.png")
        if os.path.exists(img_path):
            image_paths.append(img_path)
            valid_indices.append(i)
        else:
            logger.warning(f"Image {i}.png not found, skipping")
            pieces[i] = "empty"
    
    if not image_paths:
        return pieces
    
    # Load all images in a batch
    batch_images = []
    for img_path in image_paths:
        img = cv2.imread(img_path)
        if img is not None:
            img = cv2.resize(img, img_size)
            img = img / 255.0  # Normalize
            batch_images.append(img)
        else:
            batch_images.append(np.zeros((*img_size, 3)))  # Placeholder
    
    # Convert to numpy array
    batch_images = np.array(batch_images)
    
    # Batch prediction
    try:
        predictions = predict_batch(batch_images)
        for idx, pred in zip(valid_indices, predictions):
            predicted_class = np.argmax(pred)
            piece_name = PIECE_MAPPING.get(predicted_class, "empty")
            pieces[idx] = piece_name
            logger.debug(f"Square {idx} identified as {piece_name}")
    except Exception as e:
        logger.error(f"Error in batch prediction: {e}")
        for idx in valid_indices:
            pieces[idx] = "empty"
    
    return pieces

def map_pieces_to_board(pieces_dict, ai_color):
    """Optimized mapping function"""
    board_dict = {}
    files = "abcdefgh"
    
    for i in range(1, 65):
        row = (i - 1) // 8
        col = (i - 1) % 8
        
        if ai_color == "white":
            file_idx = col
            rank = 8 - row
        else:  # AI is black
            file_idx = 7 - col
            rank = row + 1
            
        square = f"{files[file_idx]}{rank}"
        piece = pieces_dict.get(i, "empty")
        if piece != "empty":
            board_dict[square] = piece
    
    return board_dict

def capture_and_process_board():
    """Capture screen and process board - clean storage with specific naming"""
    try:
        logger.info("Capturing screen area...")
        
        with mss.mss() as sct:
            # Get monitor dimensions
            monitor = sct.monitors[1]
            screen_width = monitor["width"]
            screen_height = monitor["height"]
            
            # Calculate capture area (override with CAPTURE_* env vars if needed)
            x1 = int(os.environ.get("CAPTURE_X", 1526))
            y1 = int(os.environ.get("CAPTURE_Y", 4))
            x2 = int(os.environ.get("CAPTURE_X2", 3070))
            y2 = int(os.environ.get("CAPTURE_Y2", 1632))
            
            # Capture area
            area = {"top": y1, "left": x1, "width": x2 - x1, "height": y2 - y1}
            screenshot = sct.grab(area)
            img = np.array(screenshot)
        
        # Save the full screenshot as 1a
        screenshot_path = os.path.join(PIECES_FOLDER, "1a.png")
        cv2.imwrite(screenshot_path, img)
        logger.info("Saved full screenshot as 1a.png")
        
        # Detect chess board using code 1 logic (this will save 1b and 1c)
        board_img = detect_chess_board(img)
        
        if board_img is None:
            logger.warning("Chess board not detected, cannot process squares")
            return False
        
        # Ensure we have a square board
        h, w = board_img.shape[:2]
        size = min(w, h)
        x_centered = (w - size) // 2
        y_centered = (h - size) // 2
        board_img = board_img[y_centered:y_centered+size, x_centered:x_centered+size]
        
        # Calculate square dimensions
        square_size = size // 8
        
        # Process all squares and save as 1.png, 2.png, ..., 64.png
        for i in range(1, 65):
            row = (i - 1) // 8
            col = (i - 1) % 8
            y1 = row * square_size
            y2 = y1 + square_size
            x1 = col * square_size
            x2 = x1 + square_size
            
            square = board_img[y1:y2, x1:x2]
            
            # Save square as i.png
            square_path = os.path.join(PIECES_FOLDER, f"{i}.png")
            cv2.imwrite(square_path, square)
        
        logger.info("Successfully processed and saved: 1a.png, 1b.png, 1c.png, and 64 squares (1.png - 64.png)")
        return True
    
    except Exception as e:
        logger.error(f"Error in capture_and_process_board: {e}")
        logger.error(str(e))
        return False

@app.route('/analyze', methods=['POST'])
def analyze_board():
    """Analyze board with clean image storage"""
    try:
        data = request.get_json(silent=True) or {}
        ai_color = data.get('aiColor', 'white')
        if ai_color not in ('white', 'black'):
            return jsonify({"error": "aiColor must be 'white' or 'black'"}), 400
        logger.info(f"Starting board analysis with AI color: {ai_color}")
        
        # Clean folder before processing
        clean_piece_folder()
        
        if not capture_and_process_board():
            return jsonify({"error": "Failed to capture and process the board"}), 500
        
        pieces_dict = identify_pieces()
        board_dict = map_pieces_to_board(pieces_dict, ai_color)
        
        logger.info(f"Analysis complete, found {len(board_dict)} pieces")
        logger.info("Saved images: 1a.png (full screenshot), 1b.png (with edges), 1c.png (cropped), 1.png-64.png (squares)")
        return jsonify(board_dict)
    
    except Exception as e:
        logger.error(f"Error in analyze_board: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    Path(PIECES_FOLDER).mkdir(parents=True, exist_ok=True)
    logger.info("Starting Flask server on http://127.0.0.1:5001")
    logger.info("Image storage: 1a.png, 1b.png, 1c.png, and 1.png-64.png")
    app.run(host='127.0.0.1', port=5001, threaded=True)

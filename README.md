# Ziffie Chess

A chess AI application that combines a **React web frontend**, a **Stockfish-powered Node.js backend**, a **computer-vision board reader** (Python + TensorFlow), and a standalone **Tkinter analysis tool**.

The project can do three things:

1. **Play chess against Stockfish** from a custom board position (web UI).
2. **Read a physical chessboard from the screen** using a trained CNN model and load that position into the web UI.
3. **Get Stockfish's best move for an arbitrary board** set up in a desktop tool.

---

## Architecture

```
+----------------------+        +---------------------------------------+
|  chess-ai (React)    |        |  backend (Node.js + Express)          |
|  - custom board setup|        |  - POST /best-move  -> Stockfish     |
|  - move highlighting |        |  - GET  /health                      |
|  - plays vs AI       |-------->   port 5000                          |
+----------------------+        +---------------------------------------+
        |
        | POST /analyze
        v
+----------------------------------------------------------+
| chess-python-backend (Python Flask)                       |
| - captures screen area (mss)                             |
| - detects + crops board (OpenCV perspective transform)    |
| - identifies each square with a CNN (TensorFlow/Keras)    |
| - returns a board dict to the frontend                    |
|   port 5001                                               |
+----------------------------------------------------------+

+----------------------------------------------------------+
| stockfish/ (standalone)                                   |
| - ziffie_chess_tool.py  Tkinter GUI: place pieces,        |
|     get best move via Stockfish (UCI)                     |
| - stockfish/             vendored Stockfish engine        |
| - test_routes.py         scratch Flask route stub (5002)  |
+----------------------------------------------------------+
```

### Ports in use

| Port | Owner                                | Purpose                     |
|------|--------------------------------------|-----------------------------|
| 3000 | `chess-ai` (React dev server)        | Web UI                      |
| 5000 | `backend/index.js` (Node.js)         | Stockfish best-move API     |
| 5001 | `cropper_backend.py` (Python Flask)  | Screen capture + board read |
| 5002 | `test_routes.py` (scratch stub)      | Route testing only          |

> **Note:** Ports 5000 and 5001 are intentionally different. Earlier versions had
> both the Node and Python servers on 5000, which made them unable to run at the
> same time — this is now fixed.

---

## Directory structure

```
ziffie_chess/
├── chess-ai/                        # React web frontend (Create React App)
│   ├── public/                      # Static HTML shell, icons, manifest
│   ├── src/
│   │   ├── App.js                   # Main chess UI (board, game logic, AI calls)
│   │   ├── index.js                 # React entry point
│   │   ├── App.css / index.css      # Styles
│   │   └── App.test.js              # Default CRA smoke test
│   ├── build/                       # Production build output
│   ├── package.json                 # Dependencies (react-chessboard, chess.js, axios)
│   └── .gitignore
│
├── backend/                         # Node.js Stockfish API
│   ├── index.js                     # Express server, /best-move + /health
│   ├── package.json
│   └── node_modules/
│
├── chess-python-backend/            # Python computer-vision backend
│   ├── cropper_backend.py           # Flask API: capture -> detect -> identify
│   ├── train.py                     # CNN training script (custom architecture)
│   ├── Brand_new_augmented_chess_dataset/   # ~67k labeled square images (13 classes)
│   ├── models/                      # Additional model storage (may be empty)
│   ├── identify/                    # Runtime images: 1a/1b/1c + 1..64 (squares)
│   ├── identif/                     # Older test captures (kept for reference)
│   ├── chess_analyzer.log           # Runtime log
│   └── *.keras / *.h5               # Trained models (see "Models" below)
│
├── stockfish/                       # Standalone desktop tool + vendored engine
│   ├── ziffie_chess_tool.py         # Tkinter GUI: board builder + best move
│   ├── test_routes.py               # Scratch Flask route stub (port 5002)
│   ├── stockfish/                   # Official Stockfish engine (README, license, .exe)
│   └── stockfish-windows-x86-64-avx2.exe   # Engine binary
│
└── stockfish-windows-x86-64-avx2.exe  # Stockfish binary (project root copy)
```

---

## Component details

### 1. Frontend — `chess-ai`

React app built with **Create React App** and [`react-chessboard`](https://github.com/Clariity/react-chessboard).

Features:
- Place any piece on any square to build a custom position before starting.
- **Remove mode** erases pieces; **Undo Placement** steps back through setup.
- Choose which color the AI plays (`AI plays as`), whether the AI moves first
  (`AI first`), and whether to use deep checkmate search (`Checkmate`).
- During a game: click your piece, see legal moves highlighted, click a destination.
- **AI turn** calls the Node backend for Stockfish's best move and plays it.
- **Analyze Board** calls the Python backend to read the board from the screen.
- Side panel tracks captured pieces, move history, and score/winner.

#### Backend URLs

The API base URLs are configurable via environment variables:

| Variable                     | Default                    |
|------------------------------|----------------------------|
| `REACT_APP_BEST_MOVE_API_URL`| `http://localhost:5000`    |
| `REACT_APP_ANALYZE_API_URL`  | `http://127.0.0.1:5001`    |

Create a `.env` file in `chess-ai/` to override them.

### 2. Stockfish API — `backend`

A minimal Express server that speaks to Stockfish over the **UCI protocol** using a child process.

- `POST /best-move` — body: `{ "fen": "...", "checkmateEnabled": bool, "analyzerMode": bool, "moveCount": int }`
  Returns `{ "bestMove": "e2e4" }`.
- `GET /health` — returns server status and the resolved Stockfish path.
- `400` if `fen` is missing; `500` on spawn/engine errors; `504` if the search exceeds the timeout.

Depth selection (per half-move count):

| Mode          | Depths (by AI move index)              | Fallback |
|---------------|----------------------------------------|----------|
| `analyzerMode`| `[12, 10, 8, 6, 5, 4, 3]`              | 2        |
| default       | `[11, 9, 7, 5, 4, 3]`                  | 2        |
| checkmate     | fixed depth `15`                       | —        |

### 3. Board reader — `chess-python-backend`

A Flask API (`cropper_backend.py`) that turns a screenshot of a chessboard into a
position object the frontend can render.

Pipeline for `POST /analyze` (body: `{ "aiColor": "white" | "black" }`):
1. **Capture** — grabs a fixed screen region with `mss`.
2. **Detect** — finds the board contour via adaptive thresholding +
   `cv2.approxPolyDP` (looks for the largest 4-corner shape with roughly square
   aspect ratio).
3. **Warp** — `four_point_transform` produces a top-down view.
4. **Crop squares** — splits the board into 64 squares, saved as `1.png..64.png`.
5. **Identify** — resizes each square to the model's expected input size and runs
   batched prediction.
6. **Map** — maps square indices back to chess coordinates
   (`a1..h8`, orientation depends on `aiColor`).

Runtime images are written to `chess-python-backend/identify/`:
- `1a.png` — raw screenshot
- `1b.png` — screenshot with the detected board outlined
- `1c.png` — perspective-corrected board
- `1..64.png` — individual squares

#### Class mapping

The CNN is trained to classify 13 classes (same order in `train.py` and `cropper_backend.py`):

```
bB  bK  bN  bP  bQ  bR  empty  wB  wK  wN  wP  wQ  wR
```

### 4. Desktop tool — `stockfish/ziffie_chess_tool.py`

A Tkinter app to build an arbitrary position and ask Stockfish for the best move.

- Place pieces by selecting a **type** (K/Q/R/B/N/P) and **color** (White/Black),
  then clicking a square. An **Eraser** mode removes pieces.
- `Get Best Move` runs the position through Stockfish (depth 12) using the UCI
  protocol and shows the result.
- Requires **both kings** on the board to generate a legal FEN.

---

## Models

Trained models live in `chess-python-backend/`:

| File                                    | Size      | Notes                         |
|-----------------------------------------|-----------|-------------------------------|
| `chess_model_efficientnet_best2.keras`  | ~52 MB    | Primary model (preferred)     |
| `chess_model_efficientnet_final2.keras` | ~52 MB    | EfficientNet variant          |
| `chess_model_improved.keras`            | ~24 MB    | Custom architecture           |
| `chess_model_improved.h5`               | ~23 MB    | Keras H5 export               |

`cropper_backend.py` looks for these in order (`.keras` first, then `.h5`, then
the `models/` folder). It reads the model's expected input size from the model
itself instead of assuming `128x128`.

### Training (`train.py`)

- Dataset: `Brand_new_augmented_chess_dataset/` — ~67,000 images across the 13
  classes (per-class counts range from ~4,000 to ~11,500).
- Image size: `64x64`, batch size `64`, up to `35` epochs, AdamW at LR `0.003`.
- Architecture: EfficientNet-inspired — depthwise separable convolution blocks,
  squeeze-and-excitation attention, global average pooling, dropout.
- Callbacks: early stopping (val_accuracy, patience 4), ReduceLROnPlateau,
  exponential LR decay.
- Augmentation: horizontal/vertical flips, rotations, brightness jitter.
- On completion it saves `ultra_chess_model_<accuracy>.h5` and a ready-to-use
  `ultra_inference.py` snippet.

---

## Getting started

### Prerequisites

- **Node.js** 18+ and npm
- **Python 3.10+** with the dependencies below
- Windows (the Stockfish binary is the `windows-x86-64-avx2` build)

### 1. Run the Node.js Stockfish backend

```bash
cd backend
npm install
node index.js
# Stockfish backend running on http://localhost:5000
```

Optional environment variables:

| Variable             | Default                                  |
|----------------------|------------------------------------------|
| `STOCKFISH_PATH`     | auto-detected (root or `stockfish/`)     |
| `PORT`               | `5000`                                   |
| `SEARCH_TIMEOUT_MS`  | `120000`                                 |

### 2. Run the Python board-reader backend

```bash
cd chess-python-backend
pip install flask flask-cors opencv-python numpy mss tensorflow pyautogui
python cropper_backend.py
# Flask server on http://127.0.0.1:5001
```

The screen-capture area is configured for a specific monitor setup. If the board
isn't detected, adjust the capture rectangle with env vars:

```bash
set CAPTURE_X=1526  # left
set CAPTURE_Y=4     # top
set CAPTURE_X2=3070 # right
set CAPTURE_Y2=1632 # bottom
python cropper_backend.py
```

### 3. Run the web frontend

```bash
cd chess-ai
npm install
npm start
# Open http://localhost:3000
```

> Keep both backends running (ports 5000 and 5001) for the full experience.

### 4. Use the desktop tool (optional)

```bash
cd stockfish
python ziffie_chess_tool.py
```

---

## Data flow — "Analyze Board"

1. You click **🔍 Analyze Board** in the web UI.
2. `chess-ai` POSTs `{ aiColor }` to `http://127.0.0.1:5001/analyze`.
3. The Python backend screenshots the region, detects and warps the board,
   and predicts every square.
4. It returns a board dictionary, e.g. `{ "a1": "wR", "b1": "wN", ... }`.
5. `chess-ai` sets that as the custom position, ready for Stockfish analysis.

## Data flow — "AI plays"

1. Click **AI plays {color}**.
2. `chess-ai` converts the current board to FEN and POSTs it to
   `http://localhost:5000/best-move`.
3. The Node backend runs Stockfish at the selected depth and returns `bestMove`.
4. `chess-ai` applies the move (with promotion handling) and updates the board.

---

## Known limitations

- **Board reader screen area is hardcoded** to one monitor layout. Tune the
  `CAPTURE_*` env vars or the defaults in `capture_and_process_board()`.
- **Frontend uses its own lightweight move generator** (in `App.js`) rather than
  `chess.js` for legal-move highlighting, so special moves like castling and
  en passant are not fully covered.
- **Winner detection is material-based** (piece-value comparison), not real
  checkmate detection. "Game over" also triggers after 12 moves.
- The **models folder** may be empty; the code falls back to the model files in
  `chess-python-backend/`.
- `test_routes.py` is a scratch file for route experiments, not part of the
  main flow.
- The trained model size/architecture assumes CPU inference; no GPU paths are
  configured.

---

## Troubleshooting

| Symptom                                | Fix                                                        |
|----------------------------------------|------------------------------------------------------------|
| `Cannot find module 'duplexer'` on build| Run `npm install` in `chess-ai/` (repairs broken node_modules) |
| Frontend says "AI service" error       | Start the Node backend (`node index.js` in `backend/`)     |
| "Analyzer failed"                      | Start the Python backend (`python cropper_backend.py`)     |
| Chessboard not detected in capture     | Check the board is fully inside the `CAPTURE_*` region     |
| `Stockfish binary not found`           | Set `STOCKFISH_PATH` to a stockfish `.exe`                 |
| Tool can't generate FEN                | Ensure both a white king and a black king are on the board |

import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, optimizers, callbacks
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import time
import gc
import warnings
warnings.filterwarnings('ignore')

# ULTRA-OPTIMIZED CONFIGURATION
class UltraConfig:
    DATA_PATH = os.environ.get(
        "CHESS_DATASET_PATH",
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "Brand_new_augmented_chess_dataset"),
    )
    IMG_SIZE = 64  # Smallest viable size for 90%+ accuracy
    BATCH_SIZE = 64  # Larger batches = faster training
    NUM_EPOCHS = 35  # Smart early stopping
    LEARNING_RATE = 0.003  # Higher LR for faster convergence
    NUM_CLASSES = 13
    CLASSES = ['bB', 'bK', 'bN', 'bP', 'bQ', 'bR', 'empty', 'wB', 'wK', 'wN', 'wP', 'wQ', 'wR']
    SEED = 42
    MAX_IMAGES_PER_CLASS = None  # Balanced dataset size
    MEMORY_EFFICIENT = True
    AGGRESSIVE_AUGMENT = True

# Configure TensorFlow for maximum CPU efficiency
print("Configuring TensorFlow for maximum CPU performance...")
tf.config.threading.set_intra_op_parallelism_threads(0)
tf.config.threading.set_inter_op_parallelism_threads(0)
tf.config.set_soft_device_placement(True)
tf.keras.mixed_precision.set_global_policy('float32')
tf.random.set_seed(UltraConfig.SEED)
np.random.seed(UltraConfig.SEED)

# MEMORY-EFFICIENT DATA GENERATOR
class MemoryEfficientGenerator(keras.utils.Sequence):
    def __init__(self, image_paths, labels, batch_size, augment=False):
        self.image_paths = image_paths
        self.labels = labels
        self.batch_size = batch_size
        self.augment = augment
        self.indexes = np.arange(len(self.image_paths))
        
    def __len__(self):
        return int(np.ceil(len(self.image_paths) / self.batch_size))
    
    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_paths = [self.image_paths[i] for i in batch_indexes]
        batch_labels = [self.labels[i] for i in batch_indexes]
        
        X, y = self._load_batch(batch_paths, batch_labels)
        return X, np.array(y)
    
    def _load_batch(self, paths, labels):
        X = np.zeros((len(paths), UltraConfig.IMG_SIZE, UltraConfig.IMG_SIZE, 3), dtype=np.float32)
        
        for i, path in enumerate(paths):
            try:
                # Ultra-fast image loading and preprocessing
                img = cv2.imread(path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = cv2.resize(img, (UltraConfig.IMG_SIZE, UltraConfig.IMG_SIZE), interpolation=cv2.INTER_LINEAR)
                
                # Aggressive augmentation for 90%+ accuracy
                if self.augment and np.random.random() > 0.3:
                    img = self._fast_augment(img)
                
                X[i] = img.astype(np.float32) / 255.0
                
            except Exception as e:
                print(f"Error loading {path}: {e}")
                X[i] = np.zeros((UltraConfig.IMG_SIZE, UltraConfig.IMG_SIZE, 3), dtype=np.float32)
        
        return X, labels
    
    def _fast_augment(self, img):
        """Lightning-fast augmentation for accuracy boost"""
        aug_type = np.random.randint(0, 6)
        
        if aug_type == 0:  # Horizontal flip
            return np.fliplr(img)
        elif aug_type == 1:  # Vertical flip
            return np.flipud(img)
        elif aug_type == 2:  # 90-degree rotation
            return np.rot90(img)
        elif aug_type == 3:  # 180-degree rotation
            return np.rot90(img, 2)
        elif aug_type == 4:  # 270-degree rotation
            return np.rot90(img, 3)
        elif aug_type == 5:  # Brightness adjustment
            factor = np.random.uniform(0.7, 1.3)
            return np.clip(img * factor, 0, 255).astype(np.uint8)
        
        return img
    
    def on_epoch_end(self):
        np.random.shuffle(self.indexes)

# ULTRA-FAST DATA LOADING
def load_data_paths():
    """Load only file paths - not actual images (memory efficient)"""
    print("Loading data paths...")
    
    image_paths = []
    labels = []
    class_to_idx = {cls: idx for idx, cls in enumerate(UltraConfig.CLASSES)}
    
    for class_name in UltraConfig.CLASSES:
        class_path = os.path.join(UltraConfig.DATA_PATH, class_name)
        if os.path.exists(class_path):
            img_files = [f for f in os.listdir(class_path) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            # Smart sampling for balanced dataset
            img_files = img_files[:UltraConfig.MAX_IMAGES_PER_CLASS]
            
            for img_name in img_files:
                img_path = os.path.join(class_path, img_name)
                image_paths.append(img_path)
                labels.append(class_to_idx[class_name])
            
            print(f"Found {len(img_files)} images for class {class_name}")
    
    return image_paths, labels

# REVOLUTIONARY ARCHITECTURE FOR 90%+ ACCURACY
def create_ultra_model():
    """Revolutionary architecture: EfficientNet-inspired with depthwise separable convolutions"""
    
    def depthwise_block(x, filters, strides=1):
        """Depthwise separable convolution block - 8x faster than regular conv"""
        x = layers.DepthwiseConv2D(3, strides=strides, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(filters, 1, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        return x
    
    def squeeze_excite_block(x, ratio=16):
        """Squeeze-and-Excitation for attention - boosts accuracy"""
        filters = x.shape[-1]
        se = layers.GlobalAveragePooling2D()(x)
        se = layers.Dense(filters // ratio, activation='relu')(se)
        se = layers.Dense(filters, activation='sigmoid')(se)
        se = layers.Reshape((1, 1, filters))(se)
        return layers.Multiply()([x, se])
    
    # Input
    inputs = layers.Input(shape=(UltraConfig.IMG_SIZE, UltraConfig.IMG_SIZE, 3))
    
    # Stem
    x = layers.Conv2D(32, 3, strides=2, padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    
    # Stage 1: 32x32
    x = depthwise_block(x, 64)
    x = squeeze_excite_block(x)
    x = layers.MaxPooling2D(2)(x)
    
    # Stage 2: 16x16
    x = depthwise_block(x, 128)
    x = squeeze_excite_block(x)
    x = layers.MaxPooling2D(2)(x)
    
    # Stage 3: 8x8
    x = depthwise_block(x, 256)
    x = squeeze_excite_block(x)
    x = layers.MaxPooling2D(2)(x)
    
    # Stage 4: 4x4
    x = depthwise_block(x, 512)
    x = squeeze_excite_block(x)
    
    # Classifier with dropout for regularization
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(UltraConfig.NUM_CLASSES, activation='softmax')(x)
    
    model = models.Model(inputs, outputs)
    return model

# LIGHTNING-FAST TRAINING PIPELINE
def ultra_fast_train():
    """Ultra-optimized training for 90%+ accuracy in minimal time"""
    
    print("=== ULTRA-OPTIMIZED CHESS RECOGNITION ===")
    start_time = time.time()
    
    # Load data paths (not actual images)
    image_paths, labels = load_data_paths()
    print(f"Total images: {len(image_paths)}")
    
    # Split data
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        image_paths, labels, test_size=0.15, stratify=labels, random_state=UltraConfig.SEED
    )
    
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_paths, train_labels, test_size=0.15, stratify=train_labels, random_state=UltraConfig.SEED
    )
    
    print(f"Train: {len(train_paths)}, Val: {len(val_paths)}, Test: {len(test_paths)}")
    
    # Create memory-efficient generators
    train_gen = MemoryEfficientGenerator(train_paths, train_labels, UltraConfig.BATCH_SIZE, augment=True)
    val_gen = MemoryEfficientGenerator(val_paths, val_labels, UltraConfig.BATCH_SIZE, augment=False)
    test_gen = MemoryEfficientGenerator(test_paths, test_labels, UltraConfig.BATCH_SIZE, augment=False)
    
    # Create ultra model
    print("Creating revolutionary model architecture...")
    model = create_ultra_model()
    
    # Ultra-optimized compilation
    model.compile(
        optimizer=optimizers.AdamW(learning_rate=UltraConfig.LEARNING_RATE, weight_decay=1e-4),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"Model parameters: {model.count_params():,}")
    
    # Smart callbacks for 90%+ accuracy
    callbacks_list = [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=4,
            restore_best_weights=True,
            verbose=1,
            min_delta=0.001
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_accuracy',
            factor=0.3,
            patience=2,
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.LearningRateScheduler(
            lambda epoch: UltraConfig.LEARNING_RATE * (0.9 ** epoch),
            verbose=0
        )
    ]
    
    # ULTRA-FAST TRAINING
    print("Starting ultra-fast training...")
    training_start = time.time()
    
    history = model.fit(
        train_gen,
        epochs=UltraConfig.NUM_EPOCHS,
        validation_data=val_gen,
        callbacks=callbacks_list,
        verbose=1, 
    )
    
    training_time = time.time() - training_start
    
    # FAST EVALUATION
    print("Evaluating model...")
    test_accuracy = model.evaluate(test_gen, verbose=1)[1]
    
    # Save ultra model
    model_name = f'ultra_chess_model_{test_accuracy:.4f}.h5'
    model.save(model_name)
    
    # Memory cleanup
    del train_gen, val_gen, test_gen
    gc.collect()
    
    # Create ultra-fast inference
    inference_code = f'''import tensorflow as tf
import numpy as np
import cv2

# Load model
model = tf.keras.models.load_model('{model_name}')
CLASSES = {UltraConfig.CLASSES}

def ultra_predict(image_path):
    """Ultra-fast prediction"""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, ({UltraConfig.IMG_SIZE}, {UltraConfig.IMG_SIZE}))
    img = np.expand_dims(img.astype(np.float32) / 255.0, 0)
    
    pred = model.predict(img, verbose=0)
    class_idx = np.argmax(pred)
    confidence = pred[0][class_idx]
    
    return CLASSES[class_idx], confidence

# Usage: piece, conf = ultra_predict('image.jpg')
'''
    
    with open('ultra_inference.py', 'w') as f:
        f.write(inference_code)
    
    total_time = time.time() - start_time
    
    # RESULTS
    print(f"\n{'='*60}")
    print(f" ULTRA-OPTIMIZATION RESULTS ")
    print(f"{'='*60}")
    print(f" TEST ACCURACY: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f" TRAINING TIME: {training_time/60:.2f} minutes")
    print(f" TOTAL TIME: {total_time/60:.2f} minutes")
    print(f" MODEL SIZE: ~{model.count_params()*4/(1024*1024):.1f} MB")
    print(f" OPTIMIZATIONS:")
    print(f"   • Image size: {UltraConfig.IMG_SIZE}x{UltraConfig.IMG_SIZE}")
    print(f"   • Batch size: {UltraConfig.BATCH_SIZE}")
    print(f"   • Depthwise separable convolutions")
    print(f"   • Squeeze-and-Excitation attention")
    print(f"   • Memory-efficient data loading")
    print(f"   • Aggressive augmentation")
    print(f"   • Smart learning rate scheduling")
    print(f" FILES SAVED:")
    print(f"   • Model: {model_name}")
    print(f"   • Inference: ultra_inference.py")
    
    if test_accuracy >= 0.90:
        print(f" SUCCESS: Achieved {test_accuracy*100:.2f}% accuracy (≥90% target)!")
    else:
        print(f"  CLOSE: {test_accuracy*100:.2f}% accuracy (target: ≥90%)")
    
    print(f"{'='*60}")
    
    return model, test_accuracy, training_time

if __name__ == "__main__":
    model, accuracy, time_taken = ultra_fast_train()
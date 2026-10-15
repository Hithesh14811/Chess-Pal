"""
Train the square classifier used by cropper_backend.py.

EfficientNetB0 (ImageNet weights) with a small dense head, trained in two
phases: first only the head with the backbone frozen, then fine-tuning the
top of the backbone at a low learning rate.

Images are fed exactly the way cropper_backend.py feeds them at inference:
OpenCV channel order (BGR), resized to 128x128, scaled to [0, 1]. Keep the
two in sync or accuracy drops for no obvious reason.

Dataset layout (one folder per class):

    Brand_new_augmented_chess_dataset/
        bB/  bK/  bN/  bP/  bQ/  bR/  empty/  wB/  wK/  wN/  wP/  wQ/  wR/

Usage:
    python train_efficientnet.py
    python train_efficientnet.py --data path/to/dataset --epochs 15 --fine-tune-epochs 15
"""

import argparse
import json
import os

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Must match PIECE_MAPPING in cropper_backend.py
CLASSES = ["bB", "bK", "bN", "bP", "bQ", "bR", "empty", "wB", "wK", "wN", "wP", "wQ", "wR"]
IMG_SIZE = 128
# Backbone layers before this index stay frozen while fine-tuning (start of block4b)
FINE_TUNE_AT = 100


def parse_args():
    p = argparse.ArgumentParser(description="Train the chess square classifier")
    p.add_argument("--data", default=os.environ.get(
        "CHESS_DATASET_PATH", os.path.join(BASE_DIR, "Brand_new_augmented_chess_dataset")))
    p.add_argument("--output", default=os.path.join(BASE_DIR, "chess_model_efficientnet_trained.keras"))
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--epochs", type=int, default=15, help="head-only epochs")
    p.add_argument("--fine-tune-epochs", type=int, default=15)
    p.add_argument("--val-split", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-pretrained", action="store_true",
                   help="random init instead of ImageNet weights (offline smoke tests)")
    return p.parse_args()


def to_model_input(images, labels):
    # image_dataset_from_directory decodes as RGB; flip to BGR like cv2.imread
    images = images[..., ::-1] / 255.0
    return images, labels


def load_datasets(args):
    common = dict(
        directory=args.data,
        labels="inferred",
        label_mode="categorical",
        class_names=CLASSES,
        image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=args.batch_size,
        validation_split=args.val_split,
        seed=args.seed,
    )
    train = tf.keras.utils.image_dataset_from_directory(subset="training", shuffle=True, **common)
    val = tf.keras.utils.image_dataset_from_directory(subset="validation", shuffle=False, **common)

    # Squares are always upright, so no flips/rotations that would turn a
    # piece upside down. Just small shifts, zoom and lighting changes, which
    # is what varies between board themes and screenshot crops.
    augment = tf.keras.Sequential([
        layers.RandomTranslation(0.08, 0.08, fill_mode="nearest"),
        layers.RandomZoom(0.1, fill_mode="nearest"),
        layers.RandomBrightness(0.15, value_range=(0, 255)),
        layers.RandomContrast(0.15),
    ])

    train = (train
             .map(lambda x, y: (augment(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
             .map(to_model_input, num_parallel_calls=tf.data.AUTOTUNE)
             .prefetch(tf.data.AUTOTUNE))
    val = val.map(to_model_input, num_parallel_calls=tf.data.AUTOTUNE).cache().prefetch(tf.data.AUTOTUNE)
    return train, val


def class_weights(data_dir):
    counts = []
    for name in CLASSES:
        folder = os.path.join(data_dir, name)
        counts.append(len(os.listdir(folder)) if os.path.isdir(folder) else 0)
    total = sum(counts)
    return {i: total / (len(CLASSES) * c) for i, c in enumerate(counts) if c}


def build_model(pretrained=True):
    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet" if pretrained else None,
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
    )
    backbone.trainable = False

    l2 = regularizers.l2(1e-4)
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        backbone,
        layers.GlobalAveragePooling2D(),
        layers.BatchNormalization(),
        layers.Dense(256, activation="relu", kernel_regularizer=l2),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(128, activation="relu", kernel_regularizer=l2),
        layers.BatchNormalization(),
        layers.Dropout(0.25),
        layers.Dense(len(CLASSES), activation="softmax"),
    ])
    return model, backbone


def compile_model(model, lr):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss="categorical_crossentropy",
        metrics=["accuracy", tf.keras.metrics.TopKCategoricalAccuracy(k=2, name="top_2_accuracy")],
    )


def callbacks(output_path):
    return [
        tf.keras.callbacks.ModelCheckpoint(output_path, monitor="val_accuracy", save_best_only=True),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-7),
    ]


def main():
    args = parse_args()
    if not os.path.isdir(args.data):
        raise SystemExit(f"Dataset not found: {args.data} (pass --data or set CHESS_DATASET_PATH)")

    tf.keras.utils.set_random_seed(args.seed)
    train, val = load_datasets(args)
    weights = class_weights(args.data)
    model, backbone = build_model(pretrained=not args.no_pretrained)

    print("Phase 1: training the head (backbone frozen)")
    compile_model(model, 1e-3)
    model.fit(train, validation_data=val, epochs=args.epochs,
              class_weight=weights, callbacks=callbacks(args.output))

    print(f"Phase 2: fine-tuning backbone layers {FINE_TUNE_AT}+")
    backbone.trainable = True
    for layer in backbone.layers[:FINE_TUNE_AT]:
        layer.trainable = False
    compile_model(model, 1e-4)
    model.fit(train, validation_data=val, epochs=args.fine_tune_epochs,
              class_weight=weights, callbacks=callbacks(args.output))

    loss, acc, top2 = model.evaluate(val, verbose=0)
    print(f"Validation accuracy: {acc:.4f}  top-2: {top2:.4f}  loss: {loss:.4f}")

    model.save(args.output)
    with open(os.path.splitext(args.output)[0] + "_metrics.json", "w") as f:
        json.dump({"val_accuracy": acc, "val_top_2_accuracy": top2, "val_loss": loss}, f, indent=2)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()

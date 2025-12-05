# file: train_ultra_lightweight_model.py
# !/usr/bin/env python3
"""
Ultra-lightweight CNN trained on HandCrop-only preprocessing.
Model saved as: best_model_ultra_light_crop_only.keras
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import tensorflow as tf
import numpy as np
from sklearn.metrics import classification_report
import os
import json

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Reproducibility
SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)


def create_ultra_lightweight_model(num_classes: int):
    """Create an ultra-lightweight CNN using Global Average Pooling."""

    model = models.Sequential([
        layers.Input(shape=(256, 256, 3)),

        # First block - 16 filters
        layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),

        # Second block - 32 filters
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        # Third block - 64 filters
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),

        # Fourth block - 128 filters
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),

        # Global Average Pooling instead of Flatten
        layers.GlobalAveragePooling2D(),

        # Compact dense layer
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),

        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])

    return model


def train_ultra_lightweight():
    """Train the ultra-lightweight model on CROP-ONLY data."""

    # === CORRECTED DATA PATHS ===
    DATA_DIR = Path("data/")
    train_dir = DATA_DIR / "processed" / "crop_only" / "train"
    val_dir   = DATA_DIR / "processed" / "crop_only" / "val"
    test_dir  = DATA_DIR / "processed" / "crop_only" / "test"

    print("\n" + "=" * 60)
    print("ULTRA-LIGHTWEIGHT MODEL TRAINING (CROP ONLY)")
    print("=" * 60)

    # Sanity checks
    for name, d in [('train', train_dir), ('val', val_dir), ('test', test_dir)]:
        if not d.exists():
            raise FileNotFoundError(f"{name} directory not found: {d}")

    # Data generators
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        rotation_range=8,
        width_shift_range=0.04,
        height_shift_range=0.04,
        zoom_range=0.04,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    val_test_datagen = ImageDataGenerator(rescale=1.0 / 255)

    batch_size = 8

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(256, 256),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True,
        seed=SEED
    )

    val_generator = val_test_datagen.flow_from_directory(
        val_dir,
        target_size=(256, 256),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    test_generator = val_test_datagen.flow_from_directory(
        test_dir,
        target_size=(256, 256),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False
    )

    # Infer number of classes
    num_classes = train_generator.num_classes
    class_indices = train_generator.class_indices

    print("\nClass indices (label mapping):")
    for cls, idx in sorted(class_indices.items(), key=lambda x: x[1]):
        print(f"  {idx}: {cls}")

    # Save mapping
    mapping_path = Path("class_indices_ultra_light_crop_only.json")
    with open(mapping_path, "w") as f:
        json.dump(class_indices, f, indent=2)
    print(f"\n💾 Saved class index mapping to: {mapping_path.resolve()}")

    # Build model
    model = create_ultra_lightweight_model(num_classes=num_classes)

    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("\nModel Summary:")
    model.summary()

    print(f"\nTotal parameters: {model.count_params():,}")

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=18,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=8,
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            "best_model_ultra_light_crop_only.keras",
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        )
    ]

    # Train
    print("\n" + "=" * 60)
    print("STARTING TRAINING (CROP ONLY)")
    print("=" * 60 + "\n")

    history = model.fit(
        train_generator,
        epochs=100,
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate
    print("\n" + "=" * 60)
    print("EVALUATION")
    print("=" * 60)

    test_loss, test_acc = model.evaluate(test_generator)
    print(f"\n📊 Test Accuracy: {test_acc:.2%}")
    print(f"📊 Test Loss: {test_loss:.4f}")

    # Predictions
    predictions = model.predict(test_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes

    # Detailed results
    print("\nDETAILED RESULTS:")
    class_names = list(test_generator.class_indices.keys())
    print(classification_report(y_true, y_pred, target_names=class_names))

    # Save history plot
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 4))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train")
    plt.plot(history.history["val_accuracy"], label="Val")
    plt.title("Model Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid(True)

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label="Val")
    plt.title("Model Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("training_history_ultra_light_crop_only.png", dpi=100)
    print("\n📊 Saved training history → training_history_ultra_light_crop_only.png")

    return model, test_acc


if __name__ == "__main__":
    model, accuracy = train_ultra_lightweight()
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE (CROP ONLY)!")
    print(f"Final Test Accuracy: {accuracy:.2%}")
    print("Model saved as: best_model_ultra_light_crop_only.keras")
    print("=" * 60)

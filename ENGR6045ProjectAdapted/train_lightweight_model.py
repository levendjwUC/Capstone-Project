# file: train_lightweight_model.py
# !/usr/bin/env python3
"""
Train a lightweight model appropriate for small datasets.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models, regularizers
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt


def create_lightweight_model(num_classes=6):
    """Create a lightweight CNN appropriate for small datasets."""

    # Much simpler architecture
    model = models.Sequential([
        layers.Input(shape=(256, 256, 3)),

        # First block - just 16 filters
        layers.Conv2D(16, (3, 3), activation='relu', padding='same',
                      kernel_regularizer=regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),

        # Second block - 32 filters
        layers.Conv2D(32, (3, 3), activation='relu', padding='same',
                      kernel_regularizer=regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),

        # Third block - 64 filters
        layers.Conv2D(64, (3, 3), activation='relu', padding='same',
                      kernel_regularizer=regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),

        # Flatten and dense layers - much smaller
        layers.Flatten(),
        layers.Dense(128, activation='relu',
                     kernel_regularizer=regularizers.l2(0.01)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),

        # Output layer
        layers.Dense(num_classes, activation='softmax')
    ])

    return model


def create_transfer_learning_model(num_classes=6):
    """Create a model using transfer learning from MobileNetV2."""

    # Use MobileNetV2 as base (lightweight and good for small datasets)
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(256, 256, 3),
        include_top=False,
        weights='imagenet'
    )

    # Freeze base model layers initially
    base_model.trainable = False

    model = models.Sequential([
        layers.Input(shape=(256, 256, 3)),

        # Data augmentation built into the model
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),

        # Preprocessing for MobileNetV2
        tf.keras.applications.mobilenet_v2.preprocess_input,

        # Base model
        base_model,

        # Custom top layers
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model


def train_model(model_type='lightweight'):
    """Train the selected model type."""

    # Data paths
    base_dir = Path('data/processed/processed_gestures_final/basic')
    train_dir = base_dir / 'train'
    val_dir = base_dir / 'val'
    test_dir = base_dir / 'test'

    print("\n" + "=" * 60)
    print(f"TRAINING {model_type.upper()} MODEL")
    print("=" * 60)

    # Data generators - less aggressive augmentation
    if model_type == 'transfer':
        # No augmentation needed as it's built into the model
        train_datagen = ImageDataGenerator(rescale=1. / 255)
    else:
        train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=10,  # Reduced from 15
            width_shift_range=0.05,  # Reduced from 0.1
            height_shift_range=0.05,  # Reduced from 0.1
            zoom_range=0.05,  # Reduced from 0.1
            horizontal_flip=True
        )

    val_test_datagen = ImageDataGenerator(rescale=1. / 255)

    # Smaller batch size for better gradient estimates
    batch_size = 8  # Reduced from 16

    # Load data
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(256, 256),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=True
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

    # Create model
    if model_type == 'transfer':
        model = create_transfer_learning_model(num_classes=6)
    else:
        model = create_lightweight_model(num_classes=6)

    # Compile with lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # Lower LR
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("\nModel Summary:")
    model.summary()

    # Count parameters
    total_params = model.count_params()
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Parameters per training sample: {total_params / 401:.0f}")

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=20,  # More patience
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=10,  # More patience
            min_lr=1e-6,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            f'best_model_{model_type}.keras',  # Use .keras format
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]

    # Train
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print(f"Training samples: {train_generator.n}")
    print(f"Validation samples: {val_generator.n}")
    print(f"Test samples: {test_generator.n}")
    print("=" * 60 + "\n")

    history = model.fit(
        train_generator,
        epochs=100,  # More epochs since model is simpler
        validation_data=val_generator,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate on test set
    print("\n" + "=" * 60)
    print("EVALUATING ON TEST SET")
    print("=" * 60)

    test_loss, test_acc = model.evaluate(test_generator)
    print(f"\nTest Accuracy: {test_acc:.2%}")
    print(f"Test Loss: {test_loss:.4f}")

    # Detailed evaluation
    predictions = model.predict(test_generator)
    y_pred = np.argmax(predictions, axis=1)
    y_true = test_generator.classes

    class_names = list(test_generator.class_indices.keys())

    print("\nCLASSIFICATION REPORT:")
    print(classification_report(y_true, y_pred, target_names=class_names))

    print("\nCONFUSION MATRIX:")
    cm = confusion_matrix(y_true, y_pred)
    print("True\\Pred", end="")
    for name in class_names:
        print(f"\t{name[:3]}", end="")
    print()
    for i, true_name in enumerate(class_names):
        print(f"{true_name[:5]}", end="")
        for j in range(len(class_names)):
            print(f"\t{cm[i, j]}", end="")
        print()

    # Plot training history
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Val Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f'training_history_{model_type}.png')
    plt.show()

    return model, history, test_acc


def main():
    """Main training function."""

    print("\n" + "=" * 60)
    print("SELECT MODEL TYPE")
    print("=" * 60)
    print("\n1. Lightweight CNN (recommended for your dataset)")
    print("2. Transfer Learning with MobileNetV2")
    print("3. Train both and compare")

    # Automatically select lightweight for now
    choice = 1  # Change this to 2 for transfer learning, or 3 for both

    results = {}

    if choice in [1, 3]:
        print("\n" + "🚀 " * 20)
        print("TRAINING LIGHTWEIGHT MODEL")
        print("🚀 " * 20)
        model, history, acc = train_model('lightweight')
        results['lightweight'] = acc

    if choice in [2, 3]:
        print("\n" + "🚀 " * 20)
        print("TRAINING TRANSFER LEARNING MODEL")
        print("🚀 " * 20)
        model, history, acc = train_model('transfer')
        results['transfer'] = acc

    # Print final summary
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE - SUMMARY")
    print("=" * 60)

    for model_name, accuracy in results.items():
        print(f"{model_name.capitalize()}: {accuracy:.2%} test accuracy")

    print("\n✅ Models saved successfully!")
    print("\nNext steps:")
    print("1. If accuracy is still low, collect more data")
    print("2. Try the transfer learning model (change choice=2)")
    print("3. Experiment with different preprocessing variants")


if __name__ == "__main__":
    # Add this to suppress TensorFlow warnings
    import os

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    main()
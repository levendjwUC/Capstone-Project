# file: train_minimal_model_fixed.py
# !/usr/bin/env python3
"""
Train a model with the correctly structured dataset.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import tensorflow as tf


def check_data_structure():
    """Check what data we have available."""
    base_dir = Path('data/processed/processed_gestures_final')

    print("\n" + "=" * 60)
    print("CHECKING DATA STRUCTURE")
    print("=" * 60)

    variants = ['basic', 'distance', 'edges', 'skeleton']
    available = {}

    for variant in variants:
        variant_dir = base_dir / variant
        if variant_dir.exists():
            splits = {}
            for split in ['train', 'val', 'test']:
                split_dir = variant_dir / split
                if split_dir.exists():
                    # Count images in this split
                    total = 0
                    classes = []
                    for class_dir in split_dir.iterdir():
                        if class_dir.is_dir():
                            images = list(class_dir.glob('*.png')) + list(class_dir.glob('*.jpg'))
                            if images:
                                total += len(images)
                                classes.append(class_dir.name)
                    splits[split] = {'total': total, 'classes': sorted(classes)}
            available[variant] = splits

    print("\nAvailable preprocessing variants:")
    for variant, splits in available.items():
        print(f"\n  {variant.upper()}:")
        for split, info in splits.items():
            if info['total'] > 0:
                print(f"    {split:5}: {info['total']:3} images across {len(info['classes'])} classes")
                if split == 'train':
                    print(f"           Classes: {', '.join(info['classes'])}")

    return available


def create_model(num_classes=6):
    """Create a simple CNN for finger counting."""
    model = models.Sequential([
        layers.Input(shape=(256, 256, 3)),

        # Block 1
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 2
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # Block 3
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),

        # Dense layers
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])

    return model


def train_with_existing_splits(variant='basic'):
    """Train using the pre-split data."""

    base_dir = Path('data/processed/processed_gestures_final') / variant

    if not base_dir.exists():
        print(f"❌ Directory not found: {base_dir}")
        return None, None

    train_dir = base_dir / 'train'
    val_dir = base_dir / 'val'
    test_dir = base_dir / 'test'

    # Check if directories exist
    for dir_path, name in [(train_dir, 'train'), (val_dir, 'val'), (test_dir, 'test')]:
        if not dir_path.exists():
            print(f"❌ {name} directory not found: {dir_path}")
            return None, None

    print(f"\n" + "=" * 60)
    print(f"TRAINING WITH '{variant.upper()}' VARIANT")
    print(f"=" * 60)
    print(f"Train directory: {train_dir}")
    print(f"Val directory: {val_dir}")
    print(f"Test directory: {test_dir}")

    # Data generators with augmentation for training
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2]
    )

    # No augmentation for validation and test
    val_test_datagen = ImageDataGenerator(rescale=1. / 255)

    # Load data
    print("\nLoading data...")

    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(256, 256),
        batch_size=16,
        class_mode='categorical',
        shuffle=True
    )

    val_generator = val_test_datagen.flow_from_directory(
        val_dir,
        target_size=(256, 256),
        batch_size=16,
        class_mode='categorical',
        shuffle=False
    )

    test_generator = val_test_datagen.flow_from_directory(
        test_dir,
        target_size=(256, 256),
        batch_size=16,
        class_mode='categorical',
        shuffle=False
    )

    # Get number of classes
    num_classes = len(train_generator.class_indices)
    print(f"\nNumber of classes detected: {num_classes}")
    print(f"Classes: {list(train_generator.class_indices.keys())}")

    # Create and compile model
    model = create_model(num_classes=num_classes)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    print("\nModel Summary:")
    model.summary()

    # Callbacks
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            verbose=1
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        tf.keras.callbacks.ModelCheckpoint(
            f'best_model_{variant}.h5',
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
        epochs=50,
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

    # Per-class accuracy
    print("\nPER-CLASS ACCURACY:")
    predictions = model.predict(test_generator)
    y_pred = tf.argmax(predictions, axis=1)
    y_true = test_generator.classes

    from sklearn.metrics import classification_report
    class_names = list(test_generator.class_indices.keys())
    print(classification_report(y_true, y_pred, target_names=class_names))

    return model, history


def main():
    """Main training function."""

    # Check available data
    available_data = check_data_structure()

    if not available_data:
        print("❌ No processed data found!")
        return

    # Choose which variant to train on
    print("\n" + "=" * 60)
    print("SELECT PREPROCESSING VARIANT")
    print("=" * 60)

    # Find which variants have data
    variants_with_data = [v for v, splits in available_data.items()
                          if 'train' in splits and splits['train']['total'] > 0]

    if not variants_with_data:
        print("❌ No variants have training data!")
        return

    print("\nAvailable variants with training data:")
    for i, variant in enumerate(variants_with_data, 1):
        train_count = available_data[variant]['train']['total']
        print(f"  {i}. {variant:<10} ({train_count} training images)")

    # For automatic selection, use 'basic' if available, otherwise first available
    if 'basic' in variants_with_data:
        selected = 'basic'
    else:
        selected = variants_with_data[0]

    print(f"\nAutomatically selected: '{selected}'")
    print("(Modify the script if you want a different variant)")

    # Train the model
    model, history = train_with_existing_splits(variant=selected)

    if model and history:
        print("\n" + "=" * 60)
        print("✅ TRAINING COMPLETE!")
        print(f"Model saved as: best_model_{selected}.h5")
        print("=" * 60)


if __name__ == "__main__":
    main()
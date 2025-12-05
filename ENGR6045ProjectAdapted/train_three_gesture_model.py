# file: train_three_gesture_model.py
# !/usr/bin/env python3
"""
Ultra-lightweight CNN for THREE gestures:
    - zero
    - one
    - five

Trained on: data/processed/crop_only/*

Saved model:
    three_gesture_model.keras

Saved class map:
    three_gesture_class_map.json
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import numpy as np
from sklearn.metrics import classification_report
import json
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)


# ------------------------------------------------------------
# Model Definition
# ------------------------------------------------------------
def create_model(num_classes: int):
    """Ultra-lightweight CNN with GAP."""
    return models.Sequential([
        layers.Input(shape=(256, 256, 3)),

        layers.Conv2D(16, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.2),

        layers.Conv2D(32, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.3),

        layers.Conv2D(64, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2),
        layers.Dropout(0.4),

        layers.Conv2D(128, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),

        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.5),

        layers.Dense(num_classes, activation='softmax')
    ])


# ------------------------------------------------------------
# Training Function
# ------------------------------------------------------------
def train_model():

    DATA = Path("data/processed/crop_only")
    TRAIN = DATA / "train"
    VAL   = DATA / "val"
    TEST  = DATA / "test"

    # Sanity check
    for name, d in [("train", TRAIN), ("val", VAL), ("test", TEST)]:
        if not d.exists():
            raise FileNotFoundError(f"❌ Missing directory: {d}")

    print("\n============================")
    print(" THREE-GESTURE MODEL TRAINING")
    print("============================")

    # ------------------------------------------
    # Generators
    # ------------------------------------------
    aug = ImageDataGenerator(
        rescale=1/255.0,
        rotation_range=8,
        width_shift_range=0.04,
        height_shift_range=0.04,
        zoom_range=0.04,
        horizontal_flip=True,
    )

    plain = ImageDataGenerator(rescale=1/255.0)
    BATCH = 8

    train_gen = aug.flow_from_directory(
        TRAIN, target_size=(256, 256), batch_size=BATCH,
        class_mode="categorical", shuffle=True, seed=SEED
    )

    val_gen = plain.flow_from_directory(
        VAL, target_size=(256, 256), batch_size=BATCH,
        class_mode="categorical", shuffle=False
    )

    test_gen = plain.flow_from_directory(
        TEST, target_size=(256, 256), batch_size=BATCH,
        class_mode="categorical", shuffle=False
    )

    # ------------------------------------------
    # Class map: folder name → index
    # ------------------------------------------
    class_map = train_gen.class_indices
    print("\nGesture mapping:")
    for k, v in class_map.items():
        print(f"  {v} = {k}")

    # Save mapping for live classifier
    with open("three_gesture_class_map.json", "w") as f:
        json.dump(class_map, f, indent=2)
    print("\n💾 Saved mapping → three_gesture_class_map.json")

    num_classes = train_gen.num_classes

    # ------------------------------------------
    # Create model
    # ------------------------------------------
    model = create_model(num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    print("\nModel Summary:")
    model.summary()

    # ------------------------------------------
    # Callbacks
    # ------------------------------------------
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            patience=12,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=6,
            min_lr=1e-6
        ),
        tf.keras.callbacks.ModelCheckpoint(
            "three_gesture_model.keras",
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1
        )
    ]

    # ------------------------------------------
    # Training
    # ------------------------------------------
    print("\n============================")
    print(" STARTING TRAINING")
    print("============================\n")

    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=60,         # Enough for 3-class, small dataset
        callbacks=callbacks,
        verbose=1
    )

    # ------------------------------------------
    # Evaluation
    # ------------------------------------------
    print("\n============================")
    print(" EVALUATION")
    print("============================")

    loss, acc = model.evaluate(test_gen)
    print(f"\n📊 Test Accuracy: {acc:.2%}")
    print(f"📊 Test Loss: {loss:.4f}")

    preds = model.predict(test_gen)
    y_pred = np.argmax(preds, axis=1)
    y_true = test_gen.classes

    print("\nClassification Report:")
    classes = list(test_gen.class_indices.keys())
    print(classification_report(y_true, y_pred, target_names=classes))

    return model, acc


# ------------------------------------------------------------
if __name__ == "__main__":
    model, acc = train_model()
    print("\n============================")
    print(" TRAINING COMPLETE!")
    print("============================")
    print(f"Final Accuracy: {acc:.2%}")
    print("Model saved as: three_gesture_model.keras")
    print("============================\n")

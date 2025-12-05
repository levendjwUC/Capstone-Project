# file: retrain_last_layer.py

import os
import sys
import json
from pathlib import Path

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

ROOT = Path(__file__).parent
ORIG_MODEL_PATH = ROOT / "three_gesture_model.keras"
RETRAIN_DIR = ROOT / "data" / "retrain"

OUTPUT_MODEL = ROOT / "three_gesture_model_retrained.keras"
OUTPUT_MAP = ROOT / "three_gesture_class_map_retrained.json"

IMG_SIZE = (256, 256)
BATCH = 12
SEED = 42


def main():

    print("\n============================")
    print("  RETRAINING LAST LAYER")
    print("============================\n")

    print(f"Loading original model: {ORIG_MODEL_PATH}")
    orig = tf.keras.models.load_model(ORIG_MODEL_PATH)

    # ---------------------------------------------------------
    # Identify the final classifier layer
    # (We assume the last layer is Dense(num_classes))
    # ---------------------------------------------------------
    if not isinstance(orig.layers[-1], layers.Dense):
        raise RuntimeError("ERROR: Last layer is NOT Dense(...). Cannot retrain head safely.")

    print(f"Original last layer: {orig.layers[-1].name}")

    # ---------------------------------------------------------
    # Clone ALL layers except final classifier
    # ---------------------------------------------------------
    base = models.Sequential()

    for layer in orig.layers[:-1]:
        base.add(layer)

    print("\nBase model without final classifier:")
    base.summary()

    # Freeze all backbone layers
    for layer in base.layers:
        layer.trainable = False

    # GAP output dimensionality
    dummy_input = tf.zeros((1, 256, 256, 3))
    dummy_out = base(dummy_input)
    feature_dim = dummy_out.shape[-1]

    print(f"\nBackbone output feature size: {feature_dim}")

    # ---------------------------------------------------------
    # Build NEW classifier head
    # ---------------------------------------------------------
    classifier = models.Sequential([
        layers.Input(shape=(feature_dim,)),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(3, activation="softmax")
    ])

    # ---------------------------------------------------------
    # Full model = base(backbone) → classifier(new head)
    # ---------------------------------------------------------
    inputs = layers.Input(shape=(256, 256, 3))
    features = base(inputs)
    outputs = classifier(features)
    final_model = models.Model(inputs=inputs, outputs=outputs)

    final_model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    # ---------------------------------------------------------
    # DATA
    # ---------------------------------------------------------
    if not RETRAIN_DIR.exists():
        raise FileNotFoundError("No retrain data found!")

    datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=6,
        width_shift_range=0.04,
        height_shift_range=0.04,
        zoom_range=0.04,
        horizontal_flip=True,
        fill_mode="nearest"
    )

    train_gen = datagen.flow_from_directory(
        RETRAIN_DIR,
        target_size=IMG_SIZE,
        batch_size=BATCH,
        class_mode="categorical",
        shuffle=True,
        seed=SEED
    )

    # Save updated class map
    with open(OUTPUT_MAP, "w") as f:
        json.dump(train_gen.class_indices, f, indent=2)
    print(f"\n💾 Saved class map → {OUTPUT_MAP}")

    # ---------------------------------------------------------
    # TRAIN
    # ---------------------------------------------------------
    print("\n🚀 Retraining classifier head...\n")

    final_model.fit(train_gen, epochs=15, verbose=1)

    # ---------------------------------------------------------
    # SAVE
    # ---------------------------------------------------------
    print(f"\n💾 Saving retrained model → {OUTPUT_MODEL}")
    final_model.save(OUTPUT_MODEL)

    print("\n============================")
    print("   RETRAINING COMPLETE")
    print("============================\n")


if __name__ == "__main__":
    main()

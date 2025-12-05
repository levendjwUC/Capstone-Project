# file: continue_training.py
# !/usr/bin/env python3
"""
Continue training from existing model with new data.
Implements fine-tuning and incremental learning strategies.
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from pathlib import Path
import numpy as np
import json
from datetime import datetime
import matplotlib.pyplot as plt
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


class IncrementalTrainer:
    """Incremental training manager for gesture recognition."""

    def __init__(self, model_path='best_model_ultra_light.keras'):
        """Initialize with existing model."""
        self.model_path = model_path
        self.model = None
        self.training_history = []

        # Load training history if exists
        self.history_file = Path('training_history.json')
        if self.history_file.exists():
            with open(self.history_file, 'r') as f:
                self.training_history = json.load(f)

    def load_and_analyze_model(self):
        """Load existing model and analyze its state."""
        print("\n" + "=" * 60)
        print("LOADING EXISTING MODEL")
        print("=" * 60)

        self.model = tf.keras.models.load_model(self.model_path)

        print(f"\n📦 Model loaded: {self.model_path}")
        print(f"📊 Total parameters: {self.model.count_params():,}")

        # Get current learning rate
        current_lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
        print(f"📈 Current learning rate: {current_lr:.6f}")

        # Model architecture summary
        print("\n🏗️ Model Architecture:")
        print(f"  Input shape: {self.model.input.shape}")
        print(f"  Output shape: {self.model.output.shape}")
        print(f"  Number of layers: {len(self.model.layers)}")

        return self.model

    def prepare_data_generators(self, fine_tuning=False):
        """Prepare data generators with appropriate augmentation."""

        # Check data locations
        processed_dir = Path('data/processed/basic')
        raw_dir = Path('data/raw')

        # Determine which data to use
        if processed_dir.exists():
            data_dir = processed_dir
            print(f"\n📁 Using processed data from: {data_dir}")
        else:
            print(f"\n⚠️ No processed data found. Please run preprocessing first.")
            return None, None, None

        # Count samples
        train_dir = data_dir / 'train'
        val_dir = data_dir / 'val'
        test_dir = data_dir / 'test'

        # Data augmentation strategy based on mode
        if fine_tuning:
            # Lighter augmentation for fine-tuning
            train_datagen = ImageDataGenerator(
                rescale=1. / 255,
                rotation_range=5,
                width_shift_range=0.05,
                height_shift_range=0.05,
                zoom_range=0.05,
                horizontal_flip=True,
                fill_mode='nearest'
            )
        else:
            # Standard augmentation for continued training
            train_datagen = ImageDataGenerator(
                rescale=1. / 255,
                rotation_range=10,
                width_shift_range=0.1,
                height_shift_range=0.1,
                zoom_range=0.1,
                horizontal_flip=True,
                brightness_range=[0.9, 1.1],
                fill_mode='nearest'
            )

        val_test_datagen = ImageDataGenerator(rescale=1. / 255)

        # Create generators
        batch_size = 16

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

        print(f"\n📊 Dataset Statistics:")
        print(f"  Training samples: {train_generator.n}")
        print(f"  Validation samples: {val_generator.n}")
        print(f"  Test samples: {test_generator.n}")

        # Check class distribution
        print(f"\n📋 Classes: {list(train_generator.class_indices.keys())}")

        return train_generator, val_generator, test_generator

    def continue_training(self, epochs=50, learning_rate=None):
        """Continue training with the existing model."""

        print("\n" + "=" * 60)
        print("CONTINUING TRAINING")
        print("=" * 60)

        # Load model
        self.load_and_analyze_model()

        # Prepare data
        train_gen, val_gen, test_gen = self.prepare_data_generators(fine_tuning=False)

        if train_gen is None:
            return None

        # Adjust learning rate if specified
        if learning_rate is not None:
            print(f"\n🔧 Setting learning rate to: {learning_rate}")
            tf.keras.backend.set_value(self.model.optimizer.learning_rate, learning_rate)
        else:
            # Use a lower learning rate for continued training
            current_lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
            new_lr = current_lr * 0.5  # Reduce by half
            print(f"\n🔧 Reducing learning rate: {current_lr:.6f} → {new_lr:.6f}")
            tf.keras.backend.set_value(self.model.optimizer.learning_rate, new_lr)

        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=20,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=10,
                min_lr=1e-7,
                verbose=1
            ),
            ModelCheckpoint(
                f'best_model_continued_{datetime.now().strftime("%Y%m%d_%H%M")}.keras',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]

        # Train
        print("\n🚀 Starting continued training...")
        print("-" * 60)

        history = self.model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )

        # Evaluate
        print("\n" + "=" * 60)
        print("EVALUATION")
        print("=" * 60)

        test_loss, test_acc = self.model.evaluate(test_gen)
        print(f"\n📊 Test Accuracy: {test_acc:.2%}")
        print(f"📊 Test Loss: {test_loss:.4f}")

        # Save training history
        session_history = {
            'timestamp': datetime.now().isoformat(),
            'epochs': epochs,
            'test_accuracy': float(test_acc),
            'test_loss': float(test_loss),
            'history': {
                'accuracy': [float(x) for x in history.history['accuracy']],
                'val_accuracy': [float(x) for x in history.history['val_accuracy']],
                'loss': [float(x) for x in history.history['loss']],
                'val_loss': [float(x) for x in history.history['val_loss']]
            }
        }

        self.training_history.append(session_history)

        with open(self.history_file, 'w') as f:
            json.dump(self.training_history, f, indent=2)

        return history, test_acc

    def fine_tune_model(self, epochs=30):
        """Fine-tune the model with new data using lower learning rate."""

        print("\n" + "=" * 60)
        print("FINE-TUNING MODEL")
        print("=" * 60)

        # Load model
        self.load_and_analyze_model()

        # Prepare data with lighter augmentation
        train_gen, val_gen, test_gen = self.prepare_data_generators(fine_tuning=True)

        if train_gen is None:
            return None

        # Use very low learning rate for fine-tuning
        fine_tune_lr = 1e-5
        print(f"\n🔧 Fine-tuning with learning rate: {fine_tune_lr}")

        # Re-compile with lower learning rate
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=fine_tune_lr),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Callbacks
        callbacks = [
            EarlyStopping(
                monitor='val_accuracy',
                patience=15,
                restore_best_weights=True,
                verbose=1
            ),
            ModelCheckpoint(
                f'best_model_finetuned_{datetime.now().strftime("%Y%m%d_%H%M")}.keras',
                monitor='val_accuracy',
                save_best_only=True,
                verbose=1
            )
        ]

        # Train
        print("\n🚀 Starting fine-tuning...")
        print("-" * 60)

        history = self.model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            callbacks=callbacks,
            verbose=1
        )

        # Evaluate
        test_loss, test_acc = self.model.evaluate(test_gen)
        print(f"\n📊 Fine-tuned Test Accuracy: {test_acc:.2%}")

        return history, test_acc

    def plot_training_progress(self):
        """Plot all training sessions to see progress over time."""

        if not self.training_history:
            print("No training history found")
            return

        plt.figure(figsize=(15, 5))

        # Plot accuracy progression
        plt.subplot(1, 3, 1)
        sessions_acc = [s['test_accuracy'] for s in self.training_history]
        sessions_dates = [s['timestamp'][:10] for s in self.training_history]
        plt.plot(sessions_acc, 'o-', linewidth=2, markersize=8)
        plt.title('Test Accuracy Over Training Sessions')
        plt.xlabel('Session')
        plt.ylabel('Accuracy')
        plt.grid(True, alpha=0.3)
        plt.ylim([0, 1])

        # Add percentage labels
        for i, acc in enumerate(sessions_acc):
            plt.text(i, acc + 0.02, f'{acc:.1%}', ha='center')

        # Plot last session's training curves
        plt.subplot(1, 3, 2)
        last_session = self.training_history[-1]
        plt.plot(last_session['history']['accuracy'], label='Train')
        plt.plot(last_session['history']['val_accuracy'], label='Val')
        plt.title('Last Session Training Curves')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)

        # Plot loss
        plt.subplot(1, 3, 3)
        plt.plot(last_session['history']['loss'], label='Train')
        plt.plot(last_session['history']['val_loss'], label='Val')
        plt.title('Last Session Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig('training_progress.png', dpi=100, bbox_inches='tight')
        plt.show()

        print(f"\n📊 Training progress saved to: training_progress.png")


def main():
    """Main training continuation function."""

    print("\n" + "🎯 " * 20)
    print("INCREMENTAL MODEL TRAINING")
    print("🎯 " * 20)

    # Check for existing models - Only ultra-light model
    model_options = [
        ('best_model_ultra_light.keras', 'Ultra-light (107K params, 76.67% acc)'),
    ]

    available_models = []
    for model_path, description in model_options:
        if Path(model_path).exists():
            available_models.append((model_path, description))

    if not available_models:
        print("❌ No trained models found!")
        return

    print("\n📦 Available models:")
    for i, (path, desc) in enumerate(available_models, 1):
        print(f"  {i}. {desc}")

    # Auto-select ultra-light (only option now)
    selected_model = available_models[0][0]
    print(f"\n✅ Selected: {selected_model}")

    # Create trainer
    trainer = IncrementalTrainer(model_path=selected_model)

    # Menu
    print("\n" + "=" * 60)
    print("TRAINING OPTIONS")
    print("=" * 60)
    print("\n1. Continue training (standard)")
    print("2. Fine-tune model (careful, low learning rate)")
    print("3. View training history")
    print("4. Evaluate current model")

    choice = input("\nSelect option (1-4): ")

    if choice == '1':
        history, acc = trainer.continue_training(epochs=50)
        if acc:
            print(f"\n🎉 Training complete! New accuracy: {acc:.2%}")
            trainer.plot_training_progress()

    elif choice == '2':
        history, acc = trainer.fine_tune_model(epochs=30)
        if acc:
            print(f"\n🎉 Fine-tuning complete! New accuracy: {acc:.2%}")

    elif choice == '3':
        trainer.plot_training_progress()

    elif choice == '4':
        trainer.load_and_analyze_model()
        _, _, test_gen = trainer.prepare_data_generators()
        if test_gen:
            loss, acc = trainer.model.evaluate(test_gen)
            print(f"\n📊 Current test accuracy: {acc:.2%}")


if __name__ == "__main__":
    main()
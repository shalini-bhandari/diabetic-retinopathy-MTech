# src/train.py

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import class_weight
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from src import config
from src.data_loader import create_data_generators
#from src.model import build_model
from src.modelB2 import build_model  # Using EfficientNetB2

def plot_history(history, history_fine):
    """Plots the full training history from both stages."""
    acc = history.history['accuracy'] + history_fine.history['accuracy']
    val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']
    loss = history.history['loss'] + history_fine.history['loss']
    val_loss = history.history['val_loss'] + history_fine.history['val_loss']

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.axvline(config.INITIAL_EPOCHS - 1, color='r', linestyle='--', label='Start Fine-Tuning')
    plt.title('Training and Validation Accuracy')
    plt.legend(loc='lower right')

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.axvline(config.INITIAL_EPOCHS - 1, color='r', linestyle='--', label='Start Fine-Tuning')
    plt.title('Training and Validation Loss')
    plt.legend(loc='upper right')
    
    plt.savefig(os.path.join(config.RESULTS_DIR, 'learning_curves.png'))
    plt.show()

def main():
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)

    train_generator, validation_generator, _ = create_data_generators()

    class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(train_generator.classes), y=train_generator.classes)
    class_weights_dict = dict(enumerate(class_weights))
    print(f"\nClass Weights: {class_weights_dict}")

    model = build_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    print("\nModel built successfully.")

    callbacks = [
        ModelCheckpoint(filepath=config.BEST_MODEL_PATH, save_best_only=True, monitor='val_accuracy', mode='max', verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, min_lr=1e-6, verbose=1)
    ]

    # --- Stage 1: Feature Extraction ---
    print("\n--- Starting Stage 1: Feature Extraction ---")
    history = model.fit(
        train_generator,
        epochs=config.INITIAL_EPOCHS,
        validation_data=validation_generator,
        steps_per_epoch=len(train_generator),
        validation_steps=len(validation_generator),
        callbacks=callbacks,
        class_weight=class_weights_dict
    )

    # --- Stage 2: Fine-Tuning ---
    print("\n--- Starting Stage 2: Fine-Tuning ---")
    
    # Extract base_model from the built model
    base_model = model.get_layer('efficientnetb2')
    base_model.trainable = True

    # Freeze the layers up to the fine-tuning point
    for layer in base_model.layers[:config.FINE_TUNE_AT_LAYER]:
        layer.trainable = False

    # Re-compile the model with a low learning rate
    model.compile(
        optimizer=Adam(learning_rate=config.LEARNING_RATE_FINE_TUNE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    total_epochs = config.INITIAL_EPOCHS + config.FINE_TUNE_EPOCHS
    history_fine = model.fit(
        train_generator,
        epochs=total_epochs,
        initial_epoch=history.epoch[-1],
        validation_data=validation_generator,
        steps_per_epoch=len(train_generator),
        validation_steps=len(validation_generator),
        callbacks=callbacks
    )

    print("\nTraining complete.")
    plot_history(history, history_fine)

if __name__ == '__main__':
    main()
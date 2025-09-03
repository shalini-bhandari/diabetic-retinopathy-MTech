# src/model.py

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from src import config

def build_model():
    """Builds the EfficientNetB0 model with a custom classification head."""
    input_shape = (config.IMG_SIZE, config.IMG_SIZE, 3)
    
    # Give the base model a unique name for easy access later
    base_model = EfficientNetB0(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet',
        name="efficientnetb0"  # <-- The important change is here
    )
    base_model.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.5)(x)
    predictions = Dense(config.NUM_CLASSES, activation='softmax')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    return model
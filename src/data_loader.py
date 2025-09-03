import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from . import config

def create_data_generators():
    # Load the CSV file
    df = pd.read_csv(config.CSV_PATH)

    df['id_code'] = df['id_code'].astype(str) + '.png'
    df['diagnosis'] = df['diagnosis'].astype(str)

    # First, split into training+validation and a separate test set (e.g., 80% train/val, 20% test)
    train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['diagnosis'])
    # Now split the remaining data into training and validation (e.g., 80% of remainder for train)
    train_df, val_df = train_test_split(train_val_df, test_size=0.2, random_state=42, stratify=train_val_df['diagnosis'])

    print(f"Total images: {len(df)}")
    print(f"Training set size: {len(train_df)}")
    print(f"Validation set size: {len(val_df)}")
    print(f"Test set size: {len(test_df)}")

    print("Setting up data generators")
    train_datagen = ImageDataGenerator(
        preprocessing_function=(lambda x: preprocess_input(x)),
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        fill_mode = 'nearest'
    )

    val_datagen = ImageDataGenerator(preprocessing_function = (lambda x: preprocess_input(x)))

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df, directory=config.SOURCE_IMAGE_DIR, x_col='id_code', y_col='diagnosis',
        target_size=(config.IMG_SIZE, config.IMG_SIZE), batch_size=config.BATCH_SIZE, class_mode='categorical'
    )
    validation_generator = val_datagen.flow_from_dataframe(
        dataframe=val_df, directory=config.SOURCE_IMAGE_DIR, x_col='id_code', y_col='diagnosis',
        target_size=(config.IMG_SIZE, config.IMG_SIZE), batch_size=config.BATCH_SIZE, class_mode='categorical'
    )
    test_generator = val_datagen.flow_from_dataframe(
        dataframe=test_df, directory=config.SOURCE_IMAGE_DIR, x_col='id_code', y_col='diagnosis',
        target_size=(config.IMG_SIZE, config.IMG_SIZE), batch_size=1, class_mode='categorical', shuffle=False
    )
    return train_generator, validation_generator, test_generator
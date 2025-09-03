# Configuration file for project paths
SOURCE_IMAGE_DIR = 'C:/Users/I768800/Documents/MTech_Project/diabetic-retinopathy-MTech/data/full_dataset/train_images'
CSV_PATH = 'C:/Users/I768800/Documents/MTech_Project/diabetic-retinopathy-MTech/data/full_dataset/train.csv'
MODELS_DIR = '/models/'
RESULTS_DIR = '/results/'

#Parameters
# IMG_SIZE = 224
# IMG_SIZE = 240 # For EfficientNetB1
IMG_SIZE = 260 # For EfficientNetB2
BATCH_SIZE = 32
NUM_CLASSES = 5
INITIAL_EPOCHS = 15
FINE_TUNE_EPOCHS = 10
TOTAL_EPOCHS = INITIAL_EPOCHS + FINE_TUNE_EPOCHS

#Training Paramenters
INITIAL_EPOCHS = 15
FINE_TUNE_EPOCHS = 15 # Increased for a more cautious approach
FINE_TUNE_AT_LAYER = 200 # Layer to unfreeze for fine-tuning EfficientNetB0
LEARNING_RATE_FINE_TUNE = 1e-5

#Model Checkpoint
BEST_MODEL_PATH = f'{MODELS_DIR}/best_model.h5'
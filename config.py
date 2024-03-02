# Path configurations
DATA_PATH = 'data/'
MODEL_SAVE_PATH = 'models/'
# config.py# config.py
class Config:
    IMAGE_DIR = 'path/to/cityscapes/images'
    MASK_DIR = 'path/to/cityscapes/masks'
    IMAGE_SIZE = (256, 256)
    BATCH_SIZE = 8
    EPOCHS = 50
    LEARNING_RATE = 1e-4
    NUM_CLASSES = 34  # Update this based on Cityscapes documentation
    TRAIN_VAL_SPLIT = 0.8
    SEED = 42

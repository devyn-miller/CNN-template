
import sys
sys.path.append('/Users/devynmiller/cpsc542_hw1_project_root/src')

from keras.preprocessing.image import ImageDataGenerator
from config import WIDTH_SHIFT_RANGE, HEIGHT_SHIFT_RANGE, HORIZONTAL_FLIP, DATA_AUGMENTATION


def create_augmentation_pipeline():
    '''Creates and returns a data augmentation pipeline using ImageDataGenerator.'''
    if DATA_AUGMENTATION:
        print("Using real-time data augmentation.")
        datagen = ImageDataGenerator(
            width_shift_range=WIDTH_SHIFT_RANGE,
            height_shift_range=HEIGHT_SHIFT_RANGE,
            horizontal_flip=HORIZONTAL_FLIP,
        )
    return datagen


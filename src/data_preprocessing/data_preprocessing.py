
from tensorflow.keras.utils import to_categorical
from config import NUM_CLASSES

def one_hot_encode(x_train, x_test, y_train, y_test):
    '''One-hot encodes the train and test labels.'''
    # Convert class vectors to binary class matrices
    y_train = to_categorical(y_train, NUM_CLASSES)
    y_test = to_categorical(y_test, NUM_CLASSES)
    # Normalize the train and test data to have values between 0 and 1.
    x_train = x_train.astype('float32') / 255
    x_test = x_test.astype('float32') / 255
    return x_train, x_test, y_train, y_test
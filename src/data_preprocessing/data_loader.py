from keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
def data():
    '''Returns train and test splits.'''
    return (x_train, y_train), (x_test, y_test)

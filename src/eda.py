import matplotlib.pyplot as plt
from keras.datasets import cifar10
import numpy as np

# Label names for CIFAR-10 dataset
label_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def data():
    '''Loads the CIFAR-10 dataset.'''
    (x_train, y_train), (x_test, y_test) = cifar10.data()
    return (x_train, y_train), (x_test, y_test)

def visualize_samples(x, y, num_samples=10):
    '''Visualizes a set of sample images from the dataset.'''
    plt.figure(figsize=(15, 15))
    for i in range(num_samples):
        idx = np.random.randint(0, x.shape[0])
        plt.subplot(num_samples // 5, 5, i + 1)
        plt.imshow(x[idx])
        plt.title(label_names[y[idx][0]])
        plt.axis('off')
    plt.show()

def class_distribution(y):
    '''Visualizes the distribution of classes in the dataset.'''
    unique, counts = np.unique(y, return_counts=True)
    plt.figure(figsize=(10, 5))
    plt.bar(unique, counts, tick_label=label_names)
    plt.title('Class Distribution in CIFAR-10 Dataset')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.show()

if __name__ == '__main__':
    (x_train, y_train), _ = data()
    visualize_samples(x_train, y_train)
    class_distribution(y_train)

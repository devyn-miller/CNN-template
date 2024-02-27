import matplotlib.pyplot as plt
import numpy as np

def plot_results(history):
    '''Plots the training and validation accuracy and loss over epochs.'''
    # Plot training & validation accuracy values
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

def display_best_worst_images(test_x, test_y, predictions, class_names):
    '''Displays the three best and three worst images from the test set.'''
    # Calculate confidence
    confidences = np.max(predictions, axis=1)

    # Calculate correctness
    correct = np.argmax(predictions, axis=1) == np.argmax(test_y, axis=1)

    # Best: Correct predictions with the highest confidence
    best_indices = np.argsort(-confidences * correct)[:3]

    # Worst: Incorrect predictions with the highest confidence
    # This computes a "wrongness" score and takes those with the highest scores
    worst_indices = np.argsort(-confidences * ~correct)[:3]

    # Display images
    plt.figure(figsize=(12, 4))
    for i, idx in enumerate(np.concatenate([best_indices, worst_indices])):
        plt.subplot(2, 3, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(test_x[idx])
        predicted_label = class_names[np.argmax(predictions[idx])]
        true_label = class_names[np.argmax(test_y[idx])]
        if i < 3:
            plt.title(f"Best #{i+1}\nPred: {predicted_label}\nTrue: {true_label}")
        else:
            plt.title(f"Worst #{i-2}\nPred: {predicted_label}\nTrue: {true_label}")

    plt.tight_layout()
    plt.show()

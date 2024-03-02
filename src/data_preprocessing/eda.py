# eda.py
import matplotlib.pyplot as plt
from data_preprocessing import get_dataset

def display_sample_images(dataset):
    plt.figure(figsize=(10, 10))
    for images, masks in dataset.take(5):  # Displaying 5 samples
        plt.subplot(5, 2, 1)
        plt.imshow(tf.keras.preprocessing.image.array_to_img(images[0]))
        plt.axis('off')
        plt.subplot(5, 2, 2)
        plt.imshow(tf.keras.preprocessing.image.array_to_img(masks[0]), cmap='gray')
        plt.axis('off')
    plt.show()

# Assuming you have a function to get a dataset
dataset = get_dataset(Config.IMAGE_DIR, Config.MASK_DIR)
display = display_sample_images(dataset)

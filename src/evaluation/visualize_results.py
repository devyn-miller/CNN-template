# visualization.py
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

def display_predictions(model, dataset, num_display=3):
    plt.figure(figsize=(15, 5 * num_display))
    
    for images, masks in dataset.take(num_display):
        pred_masks = model.predict(images)
        pred_masks = tf.argmax(pred_masks, axis=-1)
        pred_masks = pred_masks[..., tf.newaxis]
        
        for i in range(num_display):
            plt.subplot(num_display, 3, i * 3 + 1)
            plt.imshow(tf.keras.preprocessing.image.array_to_img(images[i]))
            plt.title("Input Image")
            plt.axis("off")
            
            plt.subplot(num_display, 3, i * 3 + 2)
            plt.imshow(tf.keras.preprocessing.image.array_to_img(masks[i]), cmap='gray')
            plt.title("True Mask")
            plt.axis("off")
            
            plt.subplot(num_display, 3, i * 3 + 3)
            plt.imshow(tf.keras.preprocessing.image.array_to_img(pred_masks[i]), cmap='gray')
            plt.title("Predicted Mask")
            plt.axis("off")
            
    plt.tight_layout()
    plt.show()

# Example usage
# Make sure you have a trained model and a validation dataset ready
# display_predictions(model, val_dataset, num_display=3)

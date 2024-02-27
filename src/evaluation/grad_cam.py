from tensorflow.keras.preprocessing.image import img_to_array
from tf_keras_vis.gradcam import GradcamPlusPlus
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore
from matplotlib import pyplot as plt
import numpy as np

def grad_cam_analysis(model, images, labels, class_idx=None, layer_name=None):
    '''Generates and displays Grad-CAM heatmaps for the 3 best and 3 worst images in the test set.''' 
    # Create Gradcam object
    gradcam = GradcamPlusPlus(model, model_modifier=ReplaceToLinear(), clone=False)

    # If class_idx is not specified, use the model predictions as class indices
    if class_idx is None:
        predictions = model.predict(images)
        class_idx = np.argmax(predictions, axis=1).tolist()  # Convert array to list

    # Calculate confidence scores
    confidences = np.max(predictions, axis=1)

    # Calculate correctness
    correct = np.argmax(predictions, axis=1) == np.argmax(labels, axis=1)

    # Calculate "wrongness" score
    wrongness = -confidences * ~correct

    # Get indices of the 3 best and 3 worst images
    best_indices = np.argsort(-confidences * correct)[:3]
    worst_indices = np.argsort(wrongness)[:3]

    # Generate heatmaps for all images
    heatmaps = gradcam(CategoricalScore(class_idx), images, penultimate_layer=layer_name)

    # Display the images with heatmaps for the 3 best and 3 worst images
    indices_to_display = np.concatenate([best_indices, worst_indices])
    for i, idx in enumerate(indices_to_display):
        plt.figure(figsize=(10, 5))

        # Display original image
        plt.subplot(1, 2, 1)
        plt.imshow(images[idx])
        plt.title(f'Original Image {idx+1}')
        plt.axis('off')

        # Display heatmap overlay
        heatmap = np.uint8(255 * heatmaps[idx])  # Scale heatmap to 0-255
        plt.subplot(1, 2, 2)
        plt.imshow(images[idx])
        plt.imshow(heatmap, cmap='jet', alpha=0.5)  # Overlay heatmap
        plt.title(f'Grad-CAM {idx+1}')
        plt.axis('off')

        plt.show()
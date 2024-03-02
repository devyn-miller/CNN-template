#grad_cam.py
from tf_keras_vis.gradcam import GradcamPlusPlus
from tf_keras_vis.utils.model_modifiers import ReplaceToLinear
from tf_keras_vis.utils.scores import CategoricalScore
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
import matplotlib.pyplot as plt

def grad_cam_for_segmentation(model, images, class_idx, layer_name=None):
    """
    Generates Grad-CAM heatmaps for segmentation models focusing on a specific class.
    
    Args:
    - model: The segmentation model.
    - images: A batch of images (numpy array).
    - class_idx: Index of the class to visualize.
    - layer_name: Name of the last convolutional layer. If None, automatically inferred.
    """
    # Create Gradcam object
    gradcam = GradcamPlusPlus(model,
                              model_modifier=ReplaceToLinear(),
                              clone=False)

    # Define score for the class of interest
    def score(output):
        return output[:, :, :, class_idx]
    
    # Generate heatmaps
    heatmaps = gradcam(score, images, penultimate_layer=layer_name)

    # Display the images with heatmaps
    for i, (image, heatmap) in enumerate(zip(images, heatmaps)):
        plt.figure(figsize=(10, 5))

        # Display original image
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.title(f'Original Image {i+1}')
        plt.axis('off')

        # Display heatmap overlay
        heatmap = np.uint8(255 * heatmap)  # Scale heatmap to 0-255
        plt.subplot(1, 2, 2)
        plt.imshow(image)
        plt.imshow(heatmap, cmap='jet', alpha=0.5)  # Overlay heatmap
        plt.title(f'Grad-CAM for class {class_idx}, Image {i+1}')
        plt.axis('off')

        plt.show()

# Example usage
# grad_cam_for_segmentation(model, images, class_idx=0, layer_name='last_conv_layer_name')

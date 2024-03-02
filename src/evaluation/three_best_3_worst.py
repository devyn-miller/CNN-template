#three_best_3_worst.py
def display_best_worst_predictions(model, dataset, num_examples=3):
    # Store IoU scores along with image, true mask, and predicted mask
    results = []

    for img, true_mask in dataset:
        pred_mask = model.predict(img)
        # Convert predictions to binary mask
        pred_mask_bin = tf.argmax(pred_mask, axis=-1)
        pred_mask_bin = pred_mask_bin[..., tf.newaxis]

        iou_score = tf.metrics.MeanIoU(num_classes=2)  # Adjust num_classes as necessary
        iou_score.update_state(true_mask, pred_mask_bin)
        iou = iou_score.result().numpy()
        
        results.append((iou, img, true_mask, pred_mask_bin))

    # Sort results based on IoU scores
    results.sort(key=lambda x: x[0], reverse=True)

    # Display best predictions
    plt.figure(figsize=(15, 5))
    for i in range(num_examples):
        iou, img, true_mask, pred_mask = results[i]
        plt.subplot(num_examples, 3, i * 3 + 1)
        plt.imshow(tf.keras.preprocessing.image.array_to_img(img[0]))
        plt.title("Input Image")
        plt.axis("off")

        plt.subplot(num_examples, 3, i * 3 + 2)
        plt.imshow(tf.keras.preprocessing.image.array_to_img(true_mask[0]), cmap='gray')
        plt.title(f"True Mask, IoU: {iou:.2f}")
        plt.axis("off")

        plt.subplot(num_examples, 3, i * 3 + 3)
        plt.imshow(tf.keras.preprocessing.image.array_to_img(pred_mask[0]), cmap='gray')
        plt.title("Predicted Mask")
        plt.axis("off")
    plt.tight_layout()
    plt.show()

    # Similarly, you could display the worst predictions by iterating over the last elements of the sorted `results`

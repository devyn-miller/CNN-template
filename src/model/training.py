#training.py
import tensorflow as tf
import config

def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]

def show_predictions(dataset=None, num=1):
    if dataset:
        for image, mask in dataset.take(num):
            pred_mask = model.predict(image)
            display([image[0], mask[0], create_mask(pred_mask)])
    else:
        display([sample_image, sample_mask,
                 create_mask(model.predict(sample_image[tf.newaxis, ...]))])

model = unet(input_size=(256, 256, 3), n_classes=num_classes)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Assuming `train_dataset` and `val_dataset` are created using `get_dataset`
model_history = model.fit(train_dataset, epochs=EPOCHS, validation_data=val_dataset)

# Show predictions
# show_predictions_v = show_predictions(train_dataset, 1)

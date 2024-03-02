# data_loader.py
import tensorflow as tf
import tensorflow_datasets as tfds

def load_tfds_cityscapes(batch_size):
    dataset, info = tfds.load('cityscapes', with_info=True, as_supervised=True)
    train_dataset, val_dataset = dataset['train'], dataset['validation']
    train_dataset = train_dataset.map(preprocess_image).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.map(preprocess_image).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return train_dataset, val_dataset

def preprocess_image(image, label):
    image = tf.image.resize(image, [256, 256])
    image = image / 255.0  # Normalize images to [0,1]

    label = tf.image.resize(label, [256, 256], method='nearest')
    return image, label

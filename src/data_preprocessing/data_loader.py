import tensorflow as tf
import os


# data_loading.py
import tensorflow as tf
import tensorflow_datasets as tfds

def load_tfds_cityscapes(batch_size):
    # Load Cityscapes dataset from TensorFlow Datasets
    dataset, info = tfds.load('cityscapes', with_info=True, as_supervised=True)
    # Split the dataset into train and validation
    train_dataset, val_dataset = dataset['train'], dataset['validation']
    # Preprocess and batch the datasets
    train_dataset = train_dataset.map(preprocess_image).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    val_dataset = val_dataset.map(preprocess_image).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return train_dataset, val_dataset

def parse_image(image, label):
    # Resize and normalize images, normalize masks as needed
    image = tf.image.resize(image, [256, 256])
    image = image / 255.0  # Normalize to [0,1]

    label = tf.image.resize(label, [256, 256], method='nearest')
    label = tf.cast(label, tf.float32)
    label = label / 255.0  # Normalize masks to [0,1] if necessary
    return image, label


# def parse_image(img_path, mask_path):
#     img = tf.io.read_file(img_path)
#     img = tf.image.decode_png(img, channels=3)
#     img = tf.image.resize(img, [256, 256])
#     img = img / 255.0  # Normalize to [0,1]

#     mask = tf.io.read_file(mask_path)
#     mask = tf.image.decode_png(mask, channels=1)
#     mask = tf.image.resize(mask, [256, 256], method='nearest')
#     mask = tf.cast(mask, tf.float32)
#     mask = mask / 255.0  # Normalize masks to [0,1]

#     return img, mask

def load_dataset(image_dir, mask_dir):
    image_paths = [os.path.join(image_dir, fname) for fname in os.listdir(image_dir)]
    mask_paths = [os.path.join(mask_dir, fname.replace('_leftImg8bit.png', '_gtFine_labelIds.png')) for fname in os.listdir(image_dir)]
    
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    dataset = dataset.map(parse_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    return dataset

def get_dataset(image_dir, mask_dir, batch_size):
    dataset = load_dataset(image_dir, mask_dir)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
    return dataset

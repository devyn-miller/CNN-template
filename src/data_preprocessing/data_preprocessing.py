# # Add to data_preprocessing.py

# def load_and_preprocess_data(image_dir, mask_dir):
#     dataset = get_dataset(image_dir, mask_dir)
#     dataset_size = dataset.cardinality().numpy()
#     train_size = int(dataset_size * Config.TRAIN_VAL_SPLIT)
    
#     train_dataset = dataset.take(train_size)
#     val_dataset = dataset.skip(train_size)
    
#     return train_dataset, val_dataset

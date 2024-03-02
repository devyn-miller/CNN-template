# metrics.py
import numpy as np
import tensorflow as tf

def evaluate_model(model, val_dataset):
    results = model.evaluate(val_dataset)
    print("Loss: {:.5f}, Accuracy: {:.2f}%".format(results[0], results[1] * 100))

# Assuming `val_dataset` is available
eval = evaluate_model(model, val_dataset)


from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def evaluate_model(model, x_test, y_test):
    '''Evaluates the model on the test set and prints classification metrics.'''
    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    print(classification_report(y_true, y_pred_classes))

    print(confusion_matrix(y_true, y_pred_classes))


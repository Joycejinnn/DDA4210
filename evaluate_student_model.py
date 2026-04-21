import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns


def evaluate_model(y_true, y_pred, y_probs):
    """
    Evaluate the model using multiple metrics
    
    Parameters:
        y_true: Actual labels
        y_pred: Predicted labels
        y_probs: Predicted probabilities
    """
    # Calculate evaluation metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    auc_roc = roc_auc_score(y_true, y_probs)
    cm = confusion_matrix(y_true, y_pred)

    # Print metrics
    print("Accuracy: {:.2f}%".format(accuracy * 100))
    print("F1 Score: {:.2f}".format(f1))
    print("Precision: {:.2f}".format(precision))
    print("Recall: {:.2f}".format(recall))
    print("AUC-ROC: {:.2f}".format(auc_roc))
    print("Confusion Matrix:\n", cm)
    print("Classification Report:\n", classification_report(y_true, y_pred))

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()


# Example Usage
# y_true = [0, 1, 1, 0]
# y_pred = [0, 1, 0, 0]
# y_probs = [0.1, 0.4, 0.35, 0.8]
# evaluate_model(y_true, y_pred, y_probs)

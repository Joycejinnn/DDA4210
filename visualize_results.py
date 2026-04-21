import matplotlib.pyplot as plt
import numpy as np

class ResultsVisualizer:
    def __init__(self, data):
        self.data = data

    def plot_metrics(self):
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        values = [self.data[metric] for metric in metrics]

        plt.figure(figsize=(10, 5))
        plt.bar(metrics, values, color=['blue', 'orange', 'green', 'red'])
        plt.title('Evaluation Metrics')
        plt.ylabel('Scores')
        plt.ylim(0, 1)
        plt.grid(axis='y')
        plt.show()

    def plot_confusion_matrix(self, confusion_matrix):
        plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(confusion_matrix))
        plt.xticks(tick_marks, range(len(confusion_matrix)))
        plt.yticks(tick_marks, range(len(confusion_matrix)))
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

# Example usage:
# evaluator = ResultsVisualizer({'accuracy': 0.9, 'precision': 0.85, 'recall': 0.8, 'f1_score': 0.82})
# evaluator.plot_metrics()
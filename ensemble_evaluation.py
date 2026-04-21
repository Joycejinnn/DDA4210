import joblib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

class EnsembleEvaluator:
    def __init__(self, model_paths):
        self.models = [joblib.load(path) for path in model_paths]

    def predict(self, X):
        predictions = np.array([model.predict(X) for model in self.models])
        ensemble_predictions = np.round(np.mean(predictions, axis=0)).astype(int)
        return ensemble_predictions

    def compute_metrics(self, y_true, y_pred):
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_score': f1_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred)
        }
        return metrics

    def plot_metrics(self, metrics):
        names = metrics.keys()
        values = metrics.values()

        plt.figure(figsize=(10, 5))
        plt.bar(names, values, color='skyblue')
        plt.title('Ensemble Model Evaluation Metrics')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        plt.show()

# Example usage:
# model_paths = ['model1.pkl', 'model2.pkl', 'model3.pkl']
# evaluator = EnsembleEvaluator(model_paths)
# X_test = ... # Your test features
# y_test = ... # Your true labels
# y_pred = evaluator.predict(X_test)
# metrics = evaluator.compute_metrics(y_test, y_pred)
# evaluator.plot_metrics(metrics)
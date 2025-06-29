# evaluation/evaluation.py
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluates a single model and prints metrics."""
    print(f"\nEvaluating {model_name}...")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)

    print(f"{model_name} Accuracy: {accuracy:.4f}")
    print('Classification Report:')
    print(report)
    print('Confusion Matrix:')
    print(conf_matrix)

    return accuracy, report, conf_matrix

def plot_confusion_matrix(conf_matrix, model_name, classes=['SQLInjection', 'XSS', 'Normal']):
    """Plots the confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes,
                yticklabels=classes)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'Confusion Matrix for {model_name}')
    plt.show()

def compare_model_accuracies(model_accuracies):
    """Plots a bar chart comparing model accuracies."""
    plt.figure(figsize=(10, 6))
    plt.bar(model_accuracies.keys(), model_accuracies.values())
    plt.xticks(rotation=30, ha='right')
    for key, value in model_accuracies.items():
        plt.text(key, value + 0.005, f"{value:.2f}%", ha='center', va='bottom')
    plt.xlabel('Model')
    plt.ylabel('Accuracy (%)')
    plt.title('Model Accuracy Comparison')
    plt.ylim(0, 100)
    plt.show()

def plot_actual_vs_predicted(y_test, y_pred, model_name, class_mapping={0: "SQLInjection", 1: "XSS", 2: "Normal"}):
    """Plots actual vs predicted class distribution."""
    actual_classes, actual_counts = np.unique(y_test, return_counts=True)
    actual_class_names = [class_mapping[cls] for cls in actual_classes]

    predicted_classes, predicted_counts = np.unique(y_pred, return_counts=True)
    predicted_class_names = [class_mapping[cls] for cls in predicted_classes]

    plt.figure(figsize=(10, 6))
    x_labels = actual_class_names
    width = 0.35
    plt.bar(np.arange(len(actual_classes)) - width/2, actual_counts, width=width, label="Actual")
    plt.bar(np.arange(len(predicted_classes)) + width/2, predicted_counts, width=width, label="Predicted")
    plt.xticks(ticks=np.arange(len(actual_classes)), labels=x_labels)
    plt.ylabel("Count")
    plt.title(f"Actual vs Predicted Class Comparison ({model_name})")
    plt.legend()
    plt.show()

# Example usage (assuming models, X_test, y_test are available)
# if __name__ == '__main__':
#     # Assume models = {'SVM': svm_model, 'Random Forest': rf_model, ...} and X_test, y_test are loaded
#     model_accuracy_dict = {}
#     for name, model in models.items():
#         accuracy, _, conf_matrix = evaluate_model(model, X_test, y_test, name)
#         model_accuracy_dict[name] = accuracy * 100
#         plot_confusion_matrix(conf_matrix, name)

#     compare_model_accuracies(model_accuracy_dict)

#     # For Stacker6X
#     # Assume fusion_model is trained and loaded
#     # y_pred_stacker = fusion_model.predict(X_test)
#     # plot_actual_vs_predicted(y_test, y_pred_stacker, "Stacker6X")
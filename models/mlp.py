# models/mlp.py
from sklearn.neural_network import MLPClassifier
import joblib

def train_mlp(X_train, y_train, save_path='/content/drive/MyDrive/Colab Notebooks/nn_model.pkl'):
    """Trains a Multi-layer Perceptron (MLP) Neural Network model."""
    nn_model = MLPClassifier(random_state=42, max_iter=1000)
    nn_model.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(nn_model, save_path)
    print(f"MLP Neural Network model saved to {save_path}")

    return nn_model

# Example usage (assuming X_train and y_train are available)
# if __name__ == '__main__':
#     # Assume X_train, y_train are loaded from preprocessing
#     nn_model = train_mlp(X_train, y_train)
#     print("MLP Neural Network model trained.")
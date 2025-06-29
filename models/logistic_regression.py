# models/logistic_regression.py
from sklearn.linear_model import LogisticRegression
import joblib

def train_logistic_regression(X_train, y_train, save_path='/content/drive/MyDrive/Colab Notebooks/lr_model.pkl'):
    """Trains a Logistic Regression model."""
    lr_model = LogisticRegression(random_state=42, max_iter=1000)
    lr_model.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(lr_model, save_path)
    print(f"Logistic Regression model saved to {save_path}")

    return lr_model

# Example usage (assuming X_train and y_train are available)
# if __name__ == '__main__':
#     # Assume X_train, y_train are loaded from preprocessing
#     lr_model = train_logistic_regression(X_train, y_train)
#     print("Logistic Regression model trained.")
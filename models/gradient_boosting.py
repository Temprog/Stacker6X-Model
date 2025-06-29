# models/gradient_boosting.py
from sklearn.ensemble import GradientBoostingClassifier
import joblib

def train_gradient_boosting(X_train, y_train, save_path='/content/drive/MyDrive/Colab Notebooks/gb_model.pkl'):
    """Trains a Gradient Boosting model."""
    gb_model = GradientBoostingClassifier(random_state=42)
    gb_model.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(gb_model, save_path)
    print(f"Gradient Boosting model saved to {save_path}")

    return gb_model

# Example usage (assuming X_train and y_train are available)
# if __name__ == '__main__':
#     # Assume X_train, y_train are loaded from preprocessing
#     gb_model = train_gradient_boosting(X_train, y_train)
#     print("Gradient Boosting model trained.")
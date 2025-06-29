# models/random_forest.py
from sklearn.ensemble import RandomForestClassifier
import joblib

def train_random_forest(X_train, y_train, save_path='/content/drive/MyDrive/Colab Notebooks/rf_model.pkl'):
    """Trains a Random Forest model."""
    rf_model = RandomForestClassifier(random_state=42)
    rf_model.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(rf_model, save_path)
    print(f"Random Forest model saved to {save_path}")

    return rf_model

# Example usage (assuming X_train and y_train are available)
# if __name__ == '__main__':
#     # Assume X_train, y_train are loaded from preprocessing
#     rf_model = train_random_forest(X_train, y_train)
#     print("Random Forest model trained.")
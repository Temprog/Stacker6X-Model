# models/extra_trees.py
from sklearn.ensemble import ExtraTreesClassifier
import joblib

def train_extra_trees(X_train, y_train, save_path='/content/drive/MyDrive/Colab Notebooks/et_model.pkl'):
    """Trains an Extra Trees (Extremely Randomized Trees) model."""
    et_model = ExtraTreesClassifier(random_state=42)
    et_model.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(et_model, save_path)
    print(f"Extra Trees model saved to {save_path}")

    return et_model

# Example usage (assuming X_train and y_train are available)
# if __name__ == '__main__':
#     # Assume X_train, y_train are loaded from preprocessing
#     et_model = train_extra_trees(X_train, y_train)
#     print("Extra Trees model trained.")
# models/svm_meta_model.py
from sklearn.svm import SVC
import joblib

def train_svm_meta_model(X_train, y_train, save_path='/content/drive/MyDrive/Colab Notebooks/svm_meta_model.pkl'):
    """Trains an SVM model to be used as the meta-model."""
    # Note: For stacking, the meta-model is typically trained on the out-of-fold predictions
    # of the base models. This script trains a standalone SVM.
    # In the Stacker6X class, this trained SVM instance is used as the meta-model.
    svm_model = SVC(kernel='linear', random_state=42, probability=True) # Set probability=True for predict_proba
    svm_model.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(svm_model, save_path)
    print(f"SVM meta-model saved to {save_path}")

    return svm_model

# Example usage (assuming X_train and y_train are available)
# if __name__ == '__main__':
#     # Assume X_train, y_train are loaded from preprocessing
#     svm_meta_model = train_svm_meta_model(X_train, y_train)
#     print("SVM meta-model trained.")
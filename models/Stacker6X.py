# models/Stacker6X.py
import numpy as np
from sklearn.metrics import accuracy_score, classification_report
import joblib # For saving/loading models

class Stacker6X:
    """
    Stacker6X: A custom stacking ensemble model for SQLI, XSS, and Normal detection.

    Base Models:
        - Logistic Regression (LR)
        - Neural Networks (NN)
        - Random Forest (RF)
        - Extra Trees (ET)
        - Gradient Boosting (GB)

    Meta-Model:
        - Support Vector Machine (SVM)

    Attributes:
        lr_model: Trained Logistic Regression model.
        nn_model: Trained MLP Neural Network model.
        rf_model: Trained Random Forest model.
        et_model: Trained Extra Trees model.
        gb_model: Trained Gradient Boosting model.
        meta_model: Trained SVM model (used as the meta-model).
    """

    def __init__(self, lr_model, nn_model, rf_model, et_model, gb_model, svm_meta_model):
        """
        Initializes the Stacker6X ensemble with pre-trained base models and a meta-model.

        Args:
            lr_model: Trained Logistic Regression model instance.
            nn_model: Trained MLP Neural Network model instance.
            rf_model: Trained Random Forest model instance.
            et_model: Trained Extra Trees model instance.
            gb_model: Trained Gradient Boosting model instance.
            svm_meta_model: Trained SVM model instance to be used as the meta-model.
        """
        self.lr_model = lr_model
        self.nn_model = nn_model
        self.rf_model = rf_model
        self.et_model = et_model
        self.gb_model = gb_model
        self.meta_model = svm_meta_model # Use the provided SVM as the meta-model

    def fit_meta_model(self, X_train_base_features, y_train):
        """
        Trains only the meta-model on the outputs of the base models.
        This is useful if base models are already trained and probabilities are generated.

        Args:
            X_train_base_features: Features for training the meta-model (outputs of base models).
            y_train: True labels for the training data.
        """
        print("Training meta-model...")
        self.meta_model.fit(X_train_base_features, y_train)
        print("Meta-model training complete.")


    def predict(self, X):
        """
        Makes predictions using the stacking ensemble.

        Args:
            X: Input data (e.g., TF-IDF vectorized payloads).

        Returns:
            numpy.ndarray: Predicted labels.
        """
        # Generate probabilities from base models
        # Ensure base models have predict_proba; if not, adjust their definitions
        lr_probs = self.lr_model.predict_proba(X)
        nn_probs = self.nn_model.predict_proba(X)
        rf_base_probs = self.rf_model.predict_proba(X)
        et_probs = self.et_model.predict_proba(X)
        gb_probs = self.gb_model.predict_proba(X)

        # Combine probabilities into a single feature set for the meta-model
        stacked_features = np.hstack([lr_probs, nn_probs, rf_base_probs, et_probs, gb_probs])

        # Predict with the meta-model
        return self.meta_model.predict(stacked_features)

    def evaluate(self, X_test, y_test):
        """
        Evaluates the model's performance.

        Args:
            X_test: Test input data.
            y_test: True labels for the test data.

        Returns:
            tuple: (accuracy, classification_report_string)
        """
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        return accuracy, report

    def save(self, filepath):
        """
        Saves the entire Stacker6X ensemble model using joblib.

        Args:
            filepath (str): The path to save the model file (.pkl).
        """
        joblib.dump(self, filepath)
        print(f"Stacker6X model saved to {filepath}")

    @classmethod
    def load(cls, filepath):
        """
        Loads a trained Stacker6X ensemble model from a file using joblib.

        Args:
            filepath (str): The path to the model file (.pkl).

        Returns:
            Stacker6X: The loaded Stacker6X model instance.
        """
        loaded_model = joblib.load(filepath)
        print(f"Stacker6X model loaded from {filepath}")
        return loaded_model

# Note: Base model training is expected to happen before initializing Stacker6X
# and should ideally be handled in a separate training script or function.
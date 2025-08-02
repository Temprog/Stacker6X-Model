import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import joblib

class Stacker6X:
    """
    Stacker6X: A custom stacking ensemble model.

    Base Models:
        - Logistic Regression (LR)
        - Neural Networks (NN)
        - Random Forest (RF)
        - Extra Trees (ET)
        - Gradient Boosting (GB)

    Meta-Model:
        - Reused SVM (97% Accuracy)
    """

    def __init__(self, lr_model, nn_model, rf_model, et_model, gb_model, svm_model):
        self.svm_model = None  # Exclude SVM as a base model
        self.lr_model = lr_model
        self.nn_model = nn_model
        self.rf_model = rf_model
        self.et_model = et_model
        self.gb_model = gb_model
        self.meta_model = svm_model  # Reuse the trained SVM as the meta-model

    def fit(self, X_train, y_train):
        """
        Trains the meta-model using predictions from the base models.
        """
        # Generate probabilities from base models
        lr_probs = self.lr_model.predict_proba(X_train)
        nn_probs = self.nn_model.predict_proba(X_train)
        rf_base_probs = self.rf_model.predict_proba(X_train)
        et_probs = self.et_model.predict_proba(X_train)
        gb_probs = self.gb_model.predict_proba(X_train)


        # Combine probabilities into a single feature set
        stacked_features_train = np.hstack([lr_probs, nn_probs, rf_base_probs, et_probs, gb_probs])

        # Train the meta-model (already trained SVM is reused here)
        self.meta_model.fit(stacked_features_train, y_train)

    def predict(self, X_test):
        """
        Makes predictions using the stacking ensemble.
        """
        # Generate probabilities from base models
        lr_probs = self.lr_model.predict_proba(X_test)
        nn_probs = self.nn_model.predict_proba(X_test)
        rf_base_probs = self.rf_model.predict_proba(X_test)
        et_probs = self.et_model.predict_proba(X_test)
        gb_probs = self.gb_model.predict_proba(X_test)

        # Combine probabilities into a single feature set
        stacked_features_test = np.hstack([lr_probs, nn_probs, rf_base_probs, et_probs, gb_probs])

        # Predict with the meta-model (trained SVM)
        return self.meta_model.predict(stacked_features_test)

    def evaluate(self, X_test, y_test):
        """
        Evaluates the model's performance.
        """
        y_pred = self.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        return accuracy, report

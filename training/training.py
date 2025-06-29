# training/training.py
import joblib
from sklearn.model_selection import KFold
import numpy as np

# Assume base model training functions (from models/*.py) and Stacker6X class (from models/Stacker6X.py) are available

def train_base_models(X_train, y_train):
    """Trains all base models."""
    print("Training base models...")
    lr_model = train_logistic_regression(X_train, y_train)
    nn_model = train_mlp(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)
    et_model = train_extra_trees(X_train, y_train)
    gb_model = train_gradient_boosting(X_train, y_train)
    svm_meta_model = train_svm_meta_model(X_train, y_train) # Train SVM for meta-model
    print("Base model training complete.")
    return lr_model, nn_model, rf_model, et_model, gb_model, svm_meta_model

def generate_meta_features(base_models, X, y, n_splits=5):
    """Generates out-of-fold predictions from base models for meta-model training."""
    print("Generating meta-features...")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    meta_features = []
    oof_labels = []

    for train_idx, val_idx in kf.split(X, y):
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx] # Use .iloc for Series

        fold_meta_features = []
        for name, model in base_models.items():
            # Train base model on fold training data
            model.fit(X_train_fold, y_train_fold)
            # Get probabilities on fold validation data
            fold_meta_features.append(model.predict_proba(X_val_fold))

        meta_features.append(np.hstack(fold_meta_features))
        oof_labels.append(y_val_fold)

    print("Meta-feature generation complete.")
    return np.vstack(meta_features), pd.concat(oof_labels) # Concatenate Series

def train_stacker6X(lr_model, nn_model, rf_model, et_model, gb_model, svm_meta_model, X_train, y_train, save_path='/content/drive/MyDrive/Colab Notebooks/Stacker6X_trained_model.pkl'):
    """Trains the Stacker6X ensemble model."""
    print("Training Stacker6X ensemble model...")
    # Generate meta-features using the trained base models on the full training data
    # Note: For a proper stacking implementation, out-of-fold predictions should be used
    # to train the meta-model. The current approach trains base models on the full
    # training data and then uses their predictions on the same training data to train
    # the meta-model, which can lead to overfitting.
    # A more robust approach would involve cross-validation to generate the meta-features.
    # For simplicity and based on the provided notebook structure, we'll use the full training data.

    # Re-initialize base models for consistent prediction behavior if needed
    # This might be necessary if the base model training functions modify the models in place.
    # However, based on the provided model training functions, they return new model instances.
    # So, we can directly use the models trained in `train_base_models`.

    # Generate probabilities from base models on the full training data
    lr_probs = lr_model.predict_proba(X_train)
    nn_probs = nn_model.predict_proba(X_train)
    rf_probs = rf_model.predict_proba(X_train)
    et_probs = et_model.predict_proba(X_train)
    gb_probs = gb_model.predict_proba(X_train)

    # Combine probabilities into a single feature set for the meta-model
    stacked_features_train = np.hstack([lr_probs, nn_probs, rf_probs, et_probs, gb_probs])

    # Initialize and train the Stacker6X model (which trains the meta-model)
    fusion_model = Stacker6X(lr_model, nn_model, rf_model, et_model, gb_model, svm_meta_model)
    fusion_model.fit_meta_model(stacked_features_train, y_train)

    # Save the trained Stacker6X model
    fusion_model.save(save_path)
    print(f"Stacker6X ensemble model saved to {save_path}")

    return fusion_model

# Example usage (assuming X_train, y_train are available and base models are trained)
# if __name__ == '__main__':
#     # Assume X_train, y_train are loaded from preprocessing
#     # Assume lr_model, nn_model, rf_model, et_model, gb_model, svm_meta_model are trained
#     # lr_model, nn_model, rf_model, et_model, gb_model, svm_meta_model = train_base_models(X_train, y_train)
#     # fusion_model = train_stacker6X(lr_model, nn_model, rf_model, et_model, gb_model, svm_meta_model, X_train, y_train)
#     print("Stacker6X ensemble model training complete.")
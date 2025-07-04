# deployment_simulation.py

import pandas as pd
import numpy as np
import os
import joblib
import random

# --- 1. Simulate random samples per class ---
def simulate_data(n_samples=7752):
    samples_per_class = n_samples // 3

    sql_payloads = [
        "SELECT * FROM users WHERE id=1",
        "OR 1 =1 --",
        "UNION SELECT password FROM accounts",
        "' OR '1'='1",
        "'; DROP TABLE users; --"
    ]

    xss_payloads = [
        "<script>alert('XSS')</script>",
        "<img src=x onerror=alert('XSS')>",
        "<svg onload=alert(1)>",
        "<body onload=alert('test')>",
        "<iframe src='javascript:alert(1)'></iframe>"
    ]

    normal_payloads = [
        "Normal login attempt",
        "User submitted contact form",
        "Page loaded successfully",
        "Viewing profile page",
        "Search results for 'shoes'",
        "Login successful",
        "Order placed for 3 items",
        "Welcome back, Mark!",
        "Welcome back, John!",
        "User profile updated",
        "Settings saved successfully",
        "You have logged out",
        "Search: hiking backpacks",
        "Blog post: Best coding practices",
        "Comment added: Nice article!"
    ]

    raw_text_data = (
        random.choices(sql_payloads, k=samples_per_class) +
        random.choices(xss_payloads, k=samples_per_class) +
        random.choices(normal_payloads, k=samples_per_class)
    )
    labels = [0]*samples_per_class + [1]*samples_per_class + [2]*samples_per_class

    combined = list(zip(raw_text_data, labels))
    random.shuffle(combined)
    raw_text_data, labels = zip(*combined)

    return pd.DataFrame({'text': raw_text_data, 'True_Label': labels})

# --- 2. Load model and vectorizer ---
def load_model_and_vectorizer(model_path, vectorizer_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at: {model_path}")
    if not os.path.exists(vectorizer_path):
         raise FileNotFoundError(f"Vectorizer file not found at: {vectorizer_path}")

    stacker6X_model_instance = joblib.load(model_path)
    loaded_tfidf_vectorizer = joblib.load(vectorizer_path)
    return stacker6X_model_instance, loaded_tfidf_vectorizer

# --- 3. Make predictions ---
def make_predictions(df, model, vectorizer):
    X_vectorized = vectorizer.transform(df['text'])
    y_pred = model.predict(X_vectorized)
    return y_pred

# --- 4. Add predictions to DataFrame and map to labels ---
def process_predictions(df, y_pred):
    df_result = df.copy().reset_index(drop=True)
    df_result['y_pred_st'] = y_pred

    class_mapping = {0: "SQLInjection", 1: "XSS", 2: "Normal"}
    df_result['Predicted_Label'] = [class_mapping[int(pred)] for pred in y_pred] # Ensure conversion to int

    return df_result

if __name__ == "__main__":
    # Define paths
    model_path = '/content/drive/MyDrive/Colab Notebooks/Stacker6X_trained_model.pkl'
    vectorizer_path = '/content/drive/MyDrive/Colab Notebooks/tfidf_vectorizer.pkl'

    # Simulate data
    print("Simulating data...")
    simulated_df = simulate_data()
    print(f"Simulated data shape: {simulated_df.shape}")

    # Load model and vectorizer
    print(f"Loading model from {model_path} and vectorizer from {vectorizer_path}...")
    try:
        stacker_model, tfidf_vectorizer = load_model_and_vectorizer(model_path, vectorizer_path)
        print("Model and vectorizer loaded successfully.")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit()

    # Make predictions
    print("Making predictions...")
    predictions = make_predictions(simulated_df, stacker_model, tfidf_vectorizer)
    print("Predictions made.")

    # Process predictions and display results
    print("Processing predictions...")
    result_df = process_predictions(simulated_df, predictions)
    print("\nPrediction Results (first 10 rows):")
    print(result_df[['text', 'True_Label', 'y_pred_st', 'Predicted_Label']].head(10))

    # You can further analyze the result_df here (e.g., generate classification report)
    # from sklearn.metrics import classification_report
    # print("\nClassification Report:")
    # print(classification_report(result_df['True_Label'], result_df['y_pred_st']))
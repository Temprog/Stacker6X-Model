# deployment/simulate_deployment.py
import pandas as pd
import numpy as np
import joblib
import re
import urllib.parse
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Assume Stacker6X class (from models/Stacker6X.py) is available

# Ensure NLTK resources are downloaded (same as in preprocessing.py)
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')

def load_models(stacker_model_path, tfidf_vectorizer_path):
    """Loads the trained Stacker6X model and TF-IDF vectorizer."""
    stacker6X_model = joblib.load(stacker_model_path)
    tfidf_vectorizer = joblib.load(tfidf_vectorizer_path)
    print("Models loaded successfully.")
    return stacker6X_model, tfidf_vectorizer

def preprocess_single_payload(payload, tfidf_vectorizer):
    """Preprocesses a single payload string and vectorizes it."""
    # Apply the same preprocessing steps as used during training
    payload_cleaned = str(payload).lower()
    payload_cleaned = payload_cleaned.strip().replace(r'\s+', ' ', regex=True)
    payload_cleaned = payload_cleaned.replace(r'[^\x20-\x7E]', '', regex=True)
    payload_cleaned = urllib.parse.unquote(payload_cleaned)
    payload_cleaned = re.sub(r'[^a-zA-Z0-9<>"\'=%-]', ' ', payload_cleaned)

    # Tokenization and stop word removal (using the same combined list)
    stop_words = set(stopwords.words('english'))
    custom_stop_words = {'www', 'http', 'https', 'xssed', 'xss', 'sql'}
    combined_stop_words = list(stop_words.union(custom_stop_words))
    payload_tokens = word_tokenize(payload_cleaned)
    payload_cleaned_tokens = [word for word in payload_tokens if word not in combined_stop_words]

    # Join tokens back to string for vectorization
    payload_cleaned_str = ' '.join(payload_cleaned_tokens)

    # Vectorize the cleaned payload
    X_vectorized = tfidf_vectorizer.transform([payload_cleaned_str])

    return X_vectorized

def simulate_detection(payloads, stacker6X_model, tfidf_vectorizer):
    """Simulates detection on a list of payloads."""
    predictions = []
    class_mapping = {0: "SQLInjection", 1: "XSS", 2: "Normal"}

    for payload in payloads:
        # Preprocess and vectorize the single payload
        X_payload_vectorized = preprocess_single_payload(payload, tfidf_vectorizer)

        # Make prediction using the Stacker6X model
        prediction_numeric = stacker6X_model.predict(X_payload_vectorized)[0]

        # Map numeric prediction to class name
        predicted_label = class_mapping.get(prediction_numeric, "Unknown")
        predictions.append(predicted_label)

    return predictions

# Example usage (assuming models are saved in Google Drive)
# if __name__ == '__main__':
#     stacker_model_path = '/content/drive/MyDrive/Colab Notebooks/Stacker6X_trained_model.pkl'
#     tfidf_vectorizer_path = '/content/drive/MyDrive/Colab Notebooks/tfidf_vectorizer.pkl'

#     # Load the trained models
#     stacker6X_model, tfidf_vectorizer = load_models(stacker_model_path, tfidf_vectorizer_path)

#     # Simulate new payloads
#     simulated_payloads = [
#         "SELECT * FROM users WHERE username = 'admin' --'", # SQLI
#         "<script>alert('hello')</script>", # XSS
#         "This is a normal comment.", # Normal
#         "' OR '1'='1", # SQLI
#         "<img src='x' onerror='alert(1)'>", # XSS
#         "Just a regular sentence." # Normal
#     ]

#     # Perform simulated detection
#     predictions = simulate_detection(simulated_payloads, stacker6X_model, tfidf_vectorizer)

#     # Print the results
#     print("\nSimulated Detection Results:")
#     for payload, prediction in zip(simulated_payloads, predictions):
#         print(f"Payload: '{payload}' -> Predicted: {prediction}")
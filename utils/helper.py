import pandas as pd
import numpy as np
import re
import urllib.parse
import joblib
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')


def preprocess_payload(payload):
    """
    Preprocesses a single payload string.
    """
    # Convert to lowercase
    payload = str(payload).lower()
    # Remove leading/trailing whitespaces and collapse multiple spaces
    payload = re.sub(r'\s+', ' ', payload).strip()
    # Remove non-ASCII characters
    payload = re.sub(r'[^\x20-\x7E]', '', payload)
    # Decode URL-encoded strings
    payload = urllib.parse.unquote(payload)
    # Remove special characters except SQLi or XSS detection relevant ones
    payload = re.sub(r'[^a-zA-Z0-9<>"\'=%-]', ' ', payload)
    return payload

def tokenize_payload(payload):
    """
    Tokenizes a preprocessed payload string.
    """
    return word_tokenize(payload)

def remove_stopwords(tokens):
    """
    Removes common English stopwords from a list of tokens.
    """
    stop_words = set(stopwords.words('english'))
    custom_stop_words = {'www', 'http', 'https', 'xssed', 'xss', 'sql'}
    combined_stop_words = list(stop_words.union(custom_stop_words))
    return [word for word in tokens if word not in combined_stop_words]

def load_vectorizer(vectorizer_path):
    """
    Loads a trained TF-IDF vectorizer.
    """
    return joblib.load(vectorizer_path)

def load_model(model_path):
    """
    Loads a trained machine learning model.
    """
    return joblib.load(model_path)

def predict_payload_type(payload, vectorizer, model, class_mapping):
    """
    Preprocesses, vectorizes, and predicts the type of a single payload.
    """
    preprocessed_payload = preprocess_payload(payload)
    tokenized_payload = tokenize_payload(preprocessed_payload)
    cleaned_payload = remove_stopwords(tokenized_payload)
    # TF-IDF vectorizer expects a string, so join the cleaned tokens back
    vectorized_payload = vectorizer.transform([' '.join(cleaned_payload)])
    prediction = model.predict(vectorized_payload)
    predicted_label = class_mapping[prediction[0]]
    return predicted_label

# Example usage (optional, for testing within the script)
if __name__ == '__main__':
    # Define paths (replace with your actual paths)
    vectorizer_path = '/content/drive/MyDrive/Colab Notebooks/tfidf_vectorizer.pkl'
    model_path = '/content/drive/MyDrive/Colab Notebooks/Stacker6X_trained_model.pkl' # Or your preferred model

    # Load the vectorizer and model
    try:
        loaded_vectorizer = load_vectorizer(vectorizer_path)
        loaded_model = load_model(model_path)
        class_mapping = {0: "SQLInjection", 1: "XSS", 2: "Normal"}

        # Test with some sample payloads
        sample_payloads = [
            "SELECT * FROM users WHERE id = '1'",
            "<script>alert('hello')</script>",
            "This is a normal sentence.",
            "' OR '1'='1"
        ]

        print("Testing predictions:")
        for payload in sample_payloads:
            predicted_type = predict_payload_type(payload, loaded_vectorizer, loaded_model, class_mapping)
            print(f"Payload: '{payload}' -> Predicted Type: {predicted_type}")

    except FileNotFoundError as e:
        print(f"Error loading file: {e}. Make sure the model and vectorizer paths are correct.")
    except Exception as e:
        print(f"An error occurred: {e}")

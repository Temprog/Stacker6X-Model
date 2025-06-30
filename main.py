import sys
import os

# Add the directory containing helper.py to the Python path
# This assumes helper.py is in the same directory or a subdirectory
# Adjust the path if helper.py is located elsewhere
sys.path.append('.') # Or the specific path to the directory

try:
    from helper import load_vectorizer, load_model, predict_payload_type
except ImportError:
    print("Error: Could not import functions from helper.py.")
    print("Please ensure helper.py is in the correct directory and accessible.")
    sys.exit(1)

# Define the paths to your saved vectorizer and model
# Make sure these paths are correct and the files exist
VECTORIZER_PATH = '/content/drive/MyDrive/Colab Notebooks/tfidf_vectorizer.pkl'
MODEL_PATH = '/content/drive/MyDrive/Colab Notebooks/Stacker6X_trained_model.pkl' # Or your preferred model

def main():
    """
    Main function to load the model and make predictions on sample payloads.
    """
    # Load the vectorizer and model
    try:
        vectorizer = load_vectorizer(VECTORIZER_PATH)
        model = load_model(MODEL_PATH)
        # Define the mapping from numerical labels to class names
        class_mapping = {0: "SQLInjection", 1: "XSS", 2: "Normal"}

    except FileNotFoundError as e:
        print(f"Error loading model or vectorizer: {e}")
        print("Please ensure the paths in main.py are correct and the files exist.")
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred while loading resources: {e}")
        sys.exit(1)


    print("Model and vectorizer loaded successfully.")

    # Simulate new incoming web payloads
    new_payloads = [
        "SELECT * FROM users WHERE username = 'admin' AND password = 'password'",
        "<img src='x' onerror='alert(1)'>",
        "GET /index.html HTTP/1.1",
        "'; DROP TABLE users; --",
        "<body onload=alert('test')>",
        "This is a benign comment on a blog post."
    ]

    print("\nClassifying new payloads:")
    for payload in new_payloads:
        predicted_type = predict_payload_type(payload, vectorizer, model, class_mapping)
        print(f"Payload: '{payload}'")
        print(f"Predicted Type: {predicted_type}\n")

if __name__ == "__main__":
    main()

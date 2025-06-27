# preprocessing/preprocessing.py
import pandas as pd
import numpy as np
import re
import urllib.parse
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import joblib

# Download NLTK resources if not already downloaded
try:
    nltk.data.find('tokenizers/punkt')
except nltk.downloader.DownloadError:
    nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except nltk.downloader.DownloadError:
    nltk.download('stopwords')

def load_data(filepath):
    """Loads data from a CSV file."""
    df = pd.read_csv(filepath)
    df.rename(columns={'Sentence': 'Payload'}, inplace=True)
    return df

def preprocess_data(df, sample_size=10500, random_state=42):
    """Applies preprocessing steps to the DataFrame."""
    # Sample the data
    df1 = df.sample(n=sample_size, random_state=random_state).copy()

    # Drop 'CommandInjection' column and corresponding rows
    rows_to_drop = df1['CommandInjection'] == 1
    df1 = df1[~rows_to_drop].drop(columns=['CommandInjection'])

    # Convert relevant columns to integer type
    df1['SQLInjection'] = df1['SQLInjection'].astype(int)
    df1['XSS'] = df1['XSS'].astype(int)
    df1['Normal'] = df1['Normal'].astype(int)

    # Reset index
    df1 = df1.reset_index(drop=True)

    # Convert payloads to lowercase
    df1['Payload'] = df1['Payload'].str.lower()

    # Remove leading/trailing whitespaces and collapse multiple spaces
    df1['Payload'] = df1['Payload'].str.strip().replace(r'\s+', ' ', regex=True)

    # Remove non-ASCII characters
    df1['Payload'] = df1['Payload'].replace(r'[^\x20-\x7E]', '', regex=True)

    # Decode URL-encoded strings
    df1['Payload'] = df1['Payload'].apply(urllib.parse.unquote)

    # Remove duplicate rows based on payload
    df1 = df1.drop_duplicates(subset=['Payload'])

    # Remove special characters (except relevant ones)
    df1['Payload'] = df1['Payload'].apply(lambda x: re.sub(r'[^a-zA-Z0-9<>"\'=%-]', ' ', x))

    # Tokenization
    df1['Payload_Tokens'] = df1['Payload'].apply(word_tokenize)

    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    custom_stop_words = {'www', 'http', 'https', 'xssed', 'xss', 'sql'}
    combined_stop_words = list(stop_words.union(custom_stop_words))
    df1['Payload_Cleaned'] = df1['Payload_Tokens'].apply(
        lambda tokens: [word for word in tokens if word not in combined_stop_words]
    )

    # Balance the dataset
    sql_df1 = df1[df1['SQLInjection'] == 1]
    xss_df1 = df1[df1['XSS'] == 1]
    normal_df1 = df1[df1['Normal'] == 1]

    max_size = max(len(sql_df1), len(xss_df1), len(normal_df1))

    sql_df1_upsampled = resample(sql_df1, replace=True, n_samples=max_size, random_state=random_state)
    xss_df1_upsampled = resample(xss_df1, replace=True, n_samples=max_size, random_state=random_state)
    normal_df1_upsampled = resample(normal_df1, replace=True, n_samples=max_size, random_state=random_state)

    df2 = pd.concat([sql_df1_upsampled, xss_df1_upsampled, normal_df1_upsampled])

    # Reset index again after balancing
    df2 = df2.reset_index(drop=True)

    # Feature Engineering: Add Payload_Length
    df2['Payload_Length'] = df2['Payload_Cleaned'].apply(len)

    # Feature Selection: Remove too short payloads
    df2 = df2[df2['Payload_Length'] > 3]

    # Convert Payload_Cleaned list of tokens back to string for TF-IDF
    df2['Payload_Cleaned'] = df2['Payload_Cleaned'].apply(lambda x: ' '.join(x) if isinstance(x, list) else str(x))

    return df2

def vectorize_data(df, save_path='/content/drive/MyDrive/Colab Notebooks/tfidf_vectorizer.pkl'):
    """Vectorizes the cleaned payload data using TF-IDF."""
    sample_size = df.shape[0]
    max_features = min(10500, sample_size)

    tfidf_vectorizer = TfidfVectorizer(
       stop_words='english',
       max_features=max_features,
       ngram_range=(1, 2),
       min_df=2,
       max_df=0.9
    )

    X = tfidf_vectorizer.fit_transform(df['Payload_Cleaned']).toarray()

    # Save the fitted vectorizer
    joblib.dump(tfidf_vectorizer, save_path)
    print(f"TF-IDF vectorizer saved to {save_path}")

    return X, tfidf_vectorizer

def create_labels(df):
    """Creates numerical labels from the category columns."""
    def assign_label(row):
        if row['SQLInjection'] == 1:
            return 0
        elif row['XSS'] == 1:
            return 1
        elif row['Normal'] == 1:
            return 2
        return -1

    df['Label'] = df.apply(assign_label, axis=1)
    return df['Label']

def split_data(X, y, test_size=0.3, random_state=42):
    """Splits data into training and testing sets."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test

# Example usage (assuming you have a CSV file at '/content/drive/MyDrive/Colab Notebooks/Detection_SQLI_XSS.csv')
# if __name__ == '__main__':
#     filepath = '/content/drive/MyDrive/Colab Notebooks/Detection_SQLI_XSS.csv'
#     df = load_data(filepath)
#     df_preprocessed = preprocess_data(df)
#     X, tfidf_vectorizer = vectorize_data(df_preprocessed)
#     y = create_labels(df_preprocessed)
#     X_train, X_test, y_train, y_test = split_data(X, y)
#     print("Preprocessing and data splitting complete.")
# utils/imports.py

# TensorFlow / Keras
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Masking

# NLTK for text processing
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# Scikit-learn for ML tasks
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# Visualization tools
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Ensure necessary NLTK data is downloaded (run once)
nltk.download('punkt')
nltk.download('stopwords')

# For saving and loading Python objects
import joblib

# Imports the random module for simulating/generating data randomly
import random

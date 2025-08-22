
# SQL Injection & XSS Detection Model
This project implements a machine learning pipeline to detect SQL Injection (SQLi) and Cross-Site Scripting (XSS) attacks in web request payloads, with deployment on AWS EC2 and a user-friendly frontend.



## 🚀 Overview
The project includes:

- Multiple classification models: SVM, Random Forest, Logistic Regression, Gradient Boosting, MLP, Extra Trees and a custom Stacking Ensemble (Stacker6X).
- End-to-end data preprocessing, feature engineering and vectorization.
- Visualization and evaluation of model performance.
- Deployment on AWS EC2 with a frontend interface for real-time payload testing.



## 🗂️ Dataset

- CSV file containing web request payloads and labels: SQL Injection, XSS, Command Injection, or Normal.
- Preprocessing includes sampling, cleaning, tokenization, stopword removal and balancing.



## ✨ Data Preprocessing
Key steps:
- Column renaming (Sentence → Payload).
- Dropping Command Injection data.
- Text cleaning: lowercase, remove whitespace, non-ASCII, decode URL encoding.
- Remove duplicates and irrelevant special characters.
- Tokenization and stopword removal (including custom stopwords: www, http, https, xssed, xss, sql).
- Upsampling minority classes (SQLi, XSS) to balance dataset and reduce bias.



## 🛠️ Feature Engineering & Extraction
- Added Payload_Length feature (number of tokens).
- Remove payloads with <3 tokens.
- TF-IDF vectorization (unigrams + bigrams, min_df=2, max_df=0.9, max_features based on sample size).



## 🧠 Model Architecture & Evaluation
- Models Trained: Logistic Regression, Random Forest, Gradient Boosting, Multi-layer Perceptron (MLP), Extra Trees, Support Vector Machine (SVM).
- Custom Stacking Ensemble (Stacker6X):
  - Base learners: Logistic Regression, MLP, Random Forest, Extra Trees, Gradient Boosting
  - Meta-learner: SVM
- Feature Extraction: TF-IDF with unigrams/bigrams
- Evaluation Metrics: Accuracy, classification report, confusion matrix



## 🚀 Usage & Testing
- Frontend Testing:
Users can visit the web UI (hosted on EC2) to paste a text or document and check for malicious payloads.

- Command-line Testing:
The API can also be accessed via curl for quick command-line evaluation:

```bash ```
curl -X POST https://api.stacker6.com/predict \
     -H "Content-Type: application/json" \
     -d '{"test": "type test input here"}' 
     


## 📈 Results & Impact

| Model               | Accuracy   |
| ------------------- | ---------  |
| Logistic Regression | 97.17%       |
| Random Forest       | 97.78%       |
| Gradient Boosting   | 94.45%       |
| MLP                 | 97.57%       |
| Extra Trees         | 97.63%       |
| **Stacker6X**       | **98%** ✅ |



## 🌐 Deployment & Application

### Deployment Simulation
Before deploying on EC2:
- Saved trained Stacker6X model and TF-IDF vectorizer as pickle files.

- Simulation steps:
  1. Load saved model and vectorizer.
  2. Generate synthetic payloads (SQLi, XSS, Normal).
  3. Transform payloads using TF-IDF vectorizer.
  4. Make predictions with loaded model.
  5. Compare predictions with ground truth.
✅ Ensured reliability before production deployment.



### EC2 Deployment & Frontend
Backend: Flask/FastAPI REST API serving the trained model.
Frontend: Web interface for users to paste payloads or upload .txt files.
Workflow: Payload → API → preprocessing + TF-IDF → model prediction → frontend result display.

🔗 [Live Demo](https://api.stacker6x.com) 



## 📈 Visualizations
- Word clouds (illustrating frequent terms - general, SQLi, XSS payloads).
- Pie & bar charts for class distribution.
- Model accuracy comparison charts.
- Confusion matrices for each model's prediction performance.
- Actual vs predicted counts for Stacker6X.



## 📥 How to Run the Notebook
1. Place Detection_SQLI_XSS.csv in Google Drive (/content/drive/MyDrive/Colab Notebooks/).
2. Run notebook cells sequentially.
3. Pipeline will load data, preprocess, extract features, train/evaluate models, simulate deployment.
4. Trained model and vectorizer saved as pickle files.



## ⚙️ Dependencies
pandas, numpy, os, tensorflow, nltk, sklearn, matplotlib, seaborn, wordcloud, urllib.parse, re, statsmodels, joblib
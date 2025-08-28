
# SQL Injection & XSS Detection Model
This project applies **machine learning to web security** by detecting **SQL Injection (SQLi)** and **Cross-Site Scripting (XSS)** attacks within web requests and their HTTP payloads.
It explores multiple ML models, introduces a **novel custom stacking ensemble (Stacker6X)** and is fully deployed as a **web application on AWS EC2** with both backend and frontend interfaces.


## 🚀 Overview
- Built a complete ML pipeline: data preprocessing → feature engineering → model training → evaluation → deployment.  
- Multiple classification models: SVM, Random Forest, Logistic Regression, Gradient Boosting, MLP, Extra Trees and a custom Stacking Ensemble (Stacker6X).
- End-to-end data preprocessing, feature engineering and vectorization.
- Visualization and evaluation of model performance.
- Achieved **98% accuracy** with the custom Stacker6X ensemble model.  
- Deployed on **AWS EC2** as a production-ready app for real-time prediction and payload classification with REST API serving the model and a web UI for users to test for malicious code.  


## 🌍 Usefulness & Significance
- SQL Injection (SQLi) and Cross-Site Scripting (XSS) are two of the most critical web vulnerabilities, consistently ranked in the OWASP Top 10.
- SQLi allows attackers to manipulate backend databases, while XSS enables injection of malicious scripts into web pages viewed by users.
- This project demonstrates how machine learning can be applied to strengthen web security, moving beyond traditional rule-based detection.
- By integrating the model into APIs, web frontends, or cloud deployments, it shows potential for practical intrusion detection systems (IDS) that adapt to evolving attack patterns.


## 🗂️ Dataset
- CSV file containing web request payloads and labels: SQL Injection, XSS, Command Injection, or Normal.
- Preprocessing includes sampling, cleaning, tokenization, stopword removal and balancing.
- Sourced from Kaggle with 206,636 instances.


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

```bash
curl -X POST https://api.stacker6.com/predict \
     -H "Content-Type: application/json" \
     -d '{"test": "type test input here"}' 
```     

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
- **Backend:** Flask REST API serving the trained model.
- **Frontend:** Web user interface for users to paste payloads or upload .txt files to test for malicious code.  
- **Workflow:** Payload → API → preprocessing + TF-IDF → model prediction → frontend result display.

🔗 [Live Demo](https://api.stacker6x.com) 


## 📈 Visualizations
- [Word clouds (illustrating frequent terms - SQLi, XSS payloads)](https://github.com/Temprog/Stacker6X-Model/blob/main/assets/word_cloud.png)
- [Pie & bar charts for class (categorical) distribution of the Payload](https://github.com/Temprog/Stacker6X-Model/blob/main/assets/class_distribution_charts.png)
- [Model accuracy comparison charts](https://github.com/Temprog/Stacker6X-Model/blob/main/assets/accuracy_comparison.png)
- [Confusion matrices for each model's prediction performance](https://github.com/Temprog/Stacker6X-Model/blob/main/assets/confusion_matrix.png)
- [Actual vs predicted counts for Stacker6X](https://github.com/Temprog/Stacker6X-Model/blob/main/assets/actual_predicted.png)


## 📥 How to Run the Notebook
1. Place Detection_SQLI_XSS.csv in Google Drive (/content/drive/MyDrive/Colab Notebooks/).
2. Run notebook cells sequentially.
3. Pipeline will load data, preprocess, extract features, train/evaluate models, simulate deployment.
4. Trained model and vectorizer saved as pickle files.


## 🛠️ Technologies & Tools Used
- **Languages:** Python, HTML, CSS, JavaScript  
- **ML/DL:** Scikit-learn, TensorFlow  
- **NLP:** NLTK, TF-IDF Vectorizer  
- **Visualization:** Matplotlib, Seaborn, WordCloud  
- **Deployment:** Flask/FastAPI, AWS EC2  
- **Other:** Pandas, NumPy, Statsmodels, Joblib, os, urllib.parse, re

## 🚧 Future Directions
- Expand dataset with more diverse attack types (incl. command injection).
- Explore advanced feature engineering and deep learning (RNNs, Transformers).
- Refine ensemble strategies for higher accuracy.
- Build real-time, explainable, and robust detection.
- Strengthen deployment into APIs, web apps, or WAFs.

## 📂 Related Repositories
- 🛡️ [Stacker6X Backend (Flask REST API)](https://github.com/Temprog/Stacker6X-API)  
  Flask REST API for serving predictions from the trained model, with an added regex-based guardrail to reduce false positives.  
- 🎨 [Stacker6X Frontend for API (HTML/JS UI)](https://github.com/Temprog/Stacker6X-frontend)  
  Lightweight web interface for interacting with the backend API and visualizing predictions.  

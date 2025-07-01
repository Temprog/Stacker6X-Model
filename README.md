# Development of Stacker6X: ML model for Web Vulnerability and Attack Detection (SQLi and XSS)
This project focuses on developing and evaluating a novel and advanced machine learning model calledStacker6X (an OOP-based stacked ensemble) designed to detect and classify common web attacks, specifically SQL Injection (SQLI) and Cross-Site Scripting (XSS). The goal is to identify malicious payloads within web requests targeting web applicationcs to enhance their security.

## Data Source
The dataset used for this project is sourced from the SQLi-XSS dataset available on Kaggle. It contains a collection of web request payloads labeled as either SQL Injection, XSS, Command Injection, or Normal for training and testing machine learning models. It is a collection of 206,636 data points arranged into 5 columns and it provides a comprehensive representation of web vulnerabilities relevant for developing models to detect SQLI and XSS effectively.  
Synthetic and simulated dataset with specific patterns of Payloads keywords, special characters and script tags found in SQLI and XSS was also used to test the effectiveness of the novel Stacker6X ensemble model for vulnerability detection.

## Exploratory Data Analysis (EDA)
The initial exploration of the dataset involved understanding its structure, size, and the distribution of different payload types. Key findings from the EDA include:

The dataset contains a significant number of payloads across the categories of SQL Injection, XSS, Command Injection and Normal.
Initial visualization showed the distribution of these attack types within the dataset.
The presence of special characters, keywords and patterns indicative of different attack types in the payloads was observed.
A sample of the data was used for faster processing and model experimentation.

## Data Preprocessing
The raw payload data underwent several preprocessing steps to prepare it for model training:

Sampling: A smaller sample of the dataset was initially used to improve processing speed and manageability.
Dropping Command Injection: The 'CommandInjection' column and corresponding rows were removed as this category was not included in the scope of this detection model.
Data Type Conversion: Label columns were converted to integer types.
Index Reset: The DataFrame index was reset after sampling and dropping rows to ensure proper indexing.
Lowercase Conversion: All payloads were converted to lowercase for case-insensitive analysis.
Whitespace Cleaning: Leading/trailing whitespaces were removed, and multiple spaces were collapsed.
Non-ASCII Character Removal: Non-ASCII and invisible characters were removed.
URL Decoding: URL-encoded strings were decoded to reveal the original payload content.
Duplicate Removal: Duplicate payload entries were removed to prevent bias.
Special Character Handling: Special characters were removed, with exceptions for characters relevant to SQLi and XSS detection (e.g., <, >, ', =, %, -).
Tokenization: Payloads were tokenized into words or symbols.
Stop Word Removal: Common English stop words and custom domain-specific stop words (like 'www', 'http', 'xss') were removed.
Data Balancing: The dataset was balanced using oversampling to ensure fair representation of SQL Injection, XSS, and Normal classes.
Payload Length Filtering: Payloads with fewer than 4 characters were removed as they were considered to lack meaningful impact for detection.

## Feature Engineering and Extraction
Feature engineering and extraction were crucial steps to convert the textual payload data into a numerical format suitable for machine learning models.

Payload Length: A new feature Payload_Length was created to capture the length of the cleaned payloads, which can be an   indicator of malicious intent.
TF-IDF Vectorization: The cleaned and tokenized payloads were transformed into numerical vectors using the Term Frequency-Inverse Document Frequency (TF-IDF) method. TF-IDF helps highlight terms that are important in a specific    payload but not common across all payloads, effectively capturing the significance of words or phrases for attack detection. The TF-IDF vectorizer was configured to consider unigrams and bigrams and filter out very rare or very common terms.

## Model Training and Evaluation
Several machine learning models were trained and evaluated to determine their effectiveness in classifying the web payloads. The dataset was split into training and testing sets (70/30 split) to evaluate the models on unseen data. The models trained include:

Support Vector Machine (SVM)
Random Forest
Logistic Regression
Gradient Boosting
Multi-layer Perceptron (MLP) Neural Network
Extra Trees (Extremely Randomized Trees)
Each model's performance was evaluated using standard metrics such as accuracy, precision, recall, and F1-score, and visualized using confusion matrices.

## Stack Ensemble Model (Stacker6X)
A custom stacking ensemble model, named Stacker6X, was developed to potentially improve classification performance by combining the strengths of multiple individual models.

Architecture: Stacker6X uses a stacking architecture with several base models and a meta-model.
Base Models: The base models include Random Forest, Logistic Regression, Neural Networks - Multi-Layer Perceptron (MLP), Extra Trees and Gradient Boosting. These models make predictions on the input data.
Meta-Model: A trained Support Vector Machine (SVM) was reused as the meta-model. The meta-model is trained on the predictions (specifically, the predicted probabilities) of the base models. This allows the meta-model to learn how to best combine the outputs of the base models for a final prediction.
Performance: The Stacker6X model demonstrated strong performance, often achieving accuracy comparable to or exceeding the best individual base models.

## Deployment Helper Script (helper.py)
A Python script named helper.py was created to facilitate the deployment of the trained model for classifying new, incoming web payloads. This script encapsulates the necessary preprocessing steps and model loading/prediction logic.

Functionality: It includes functions to preprocess a raw payload, load the saved TF-IDF vectorizer and the trained Stacker6X model, and a main function to take a new payload and return its predicted class (SQL Injection, XSS or Normal).
Reusability: This script allows for easy integration of the trained model into other applications or systems for real-time or batch payload classification without needing the entire training notebook environment.

## How to Run the Notebook
To run this notebook and reproduce the results:

Clone the Repository: If this notebook is part of a GitHub repository, clone the repository to your local machine or Google Drive.
Open in Google Colab: Upload the notebook to Google Colab or open it directly from Google Drive.
Mount Google Drive: Ensure you mount your Google Drive to access the dataset and saved model/vectorizer files.
Install Dependencies: Run the cell containing !pip install and nltk.download() commands to install the necessary libraries.
Execute Cells Sequentially: Run the code cells sequentially from top to bottom. Follow the markdown headings to understand the different stages of the project.
Ensure File Paths are Correct: Verify that the file paths for the dataset and the saved model/vectorizer (/content/drive/MyDrive/Colab Notebooks/...) are correct for your environment.

## Future Improvements
Several areas can be explored for future improvements:

Larger Dataset: Training on a larger and more diverse dataset would likely improve the model's generalization capabilities.
More Advanced Preprocessing: Experimenting with other text preprocessing techniques, such as stemming, lemmatization or more sophisticated noise reduction.
Different Feature Engineering: Exploring other feature engineering methods beyond TF-IDF, such as word embeddings (Word2Vec, GloVe) or character-level features. Payload lengths can can also be explore as features.
Hyperparameter Tuning: Performing more extensive hyperparameter tuning for the base models and the meta-model in the stacking ensemble.
Other Ensemble Techniques: Investigating other ensemble methods like bagging, boosting (beyond Gradient Boosting), or voting classifiers.
Deep Learning Models: Exploring the use of deep learning models, such as Convolutional Neural Networks (CNNs) or Recurrent Neural Networks (RNNs), which are often effective for sequence data like text.
Real-time Deployment: Implementing a more robust real-time deployment solution for the trained model (e.g., using Flask, FastAPI, or cloud-based services).
Handling Command Injection: Expanding the model to also detect and classify Command Injection attacks.

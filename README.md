# Sentiment-Analysis-on-ChatGPT-Tweets
Project overview
This application analyzes text related to ChatGPT and predicts its sentiment using supervised machine-learning classification.

Problem addressed
Large volumes of user comments and reviews are difficult to inspect manually. The application provides a repeatable way to classify text into the sentiment labels learned from the training dataset.

Workflow
Upload a labeled dataset containing tweets and labels.
Remove duplicates and records with missing labels or empty text.
Normalize text by removing URLs, usernames, HTML tags and punctuation.
Convert cleaned text into TF-IDF features.
Train Logistic Regression and Multinomial Naive Bayes classifiers.
Compare accuracy and macro F1, then predict sentiment for new text.
Technology stack
Python • Streamlit • Pandas • Scikit-learn • TF-IDF • Matplotlib • Joblib

Note: The notebook uses NLTK English stopwords. This deployable app uses Scikit-learn's English stopword list (while retaining negation words) to avoid a runtime NLTK download. For exact parity, use the same stopword list in training and inference.

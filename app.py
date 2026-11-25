import re
import io
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

st.set_page_config(
    page_title="ChatGPT Sentiment Analyzer",
    page_icon="💬",
    layout="wide"
)

# Keep negation words because they can reverse sentiment.
STOP_WORDS = set(ENGLISH_STOP_WORDS)
STOP_WORDS.difference_update({"not", "no", "nor", "never"})

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = text.replace("#", "")
    text = re.sub(r"<.*?>", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    tokens = text.split()
    tokens = [word for word in tokens if word not in STOP_WORDS]
    return " ".join(tokens)

def train_models(data):
    required = {"tweets", "labels"}
    if not required.issubset(data.columns):
        missing = ", ".join(sorted(required - set(data.columns)))
        raise ValueError(f"CSV must contain these columns: tweets and labels. Missing: {missing}")

    df = data.copy()
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    df = df.drop_duplicates()
    df = df.dropna(subset=["labels"])
    df["tweets"] = df["tweets"].fillna("").astype(str)
    df = df[df["tweets"].str.strip() != ""].copy()
    df["labels"] = df["labels"].astype(str)

    if df["labels"].nunique() < 2:
        raise ValueError("At least two different sentiment labels are required.")
    if df["labels"].value_counts().min() < 2:
        raise ValueError("Each sentiment class needs at least two rows for a stratified train/test split.")

    df["clean_tweets"] = df["tweets"].apply(clean_text)
    X, y = df["clean_tweets"], df["labels"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    vectorizer = TfidfVectorizer(
        max_features=20000, ngram_range=(1, 2),
        min_df=2, sublinear_tf=True
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    lr = LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42
    )
    nb = MultinomialNB()
    lr.fit(X_train_vec, y_train)
    nb.fit(X_train_vec, y_train)

    lr_pred = lr.predict(X_test_vec)
    nb_pred = nb.predict(X_test_vec)

    metrics = {
        "Logistic Regression": {
            "accuracy": accuracy_score(y_test, lr_pred),
            "macro_f1": f1_score(y_test, lr_pred, average="macro"),
            "predictions": lr_pred
        },
        "Naive Bayes": {
            "accuracy": accuracy_score(y_test, nb_pred),
            "macro_f1": f1_score(y_test, nb_pred, average="macro"),
            "predictions": nb_pred
        }
    }
    return {
        "df": df, "vectorizer": vectorizer, "lr": lr, "nb": nb,
        "X_test": X_test, "y_test": y_test, "metrics": metrics
    }

def predict_one(text, model, vectorizer):
    cleaned = clean_text(text)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]
    probabilities = model.predict_proba(vector)[0]
    return cleaned, prediction, probabilities, model.classes_

def load_saved_artifacts():
    model_path, vectorizer_path = "sentiment_model.pkl", "tfidf_vectorizer.pkl"
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        return joblib.load(model_path), joblib.load(vectorizer_path)
    return None, None

st.title("💬 ChatGPT Sentiment Analysis")
st.caption("A machine-learning application that classifies text into sentiment categories using TF-IDF and Logistic Regression.")

with st.sidebar:
    st.header("⚙️ Model setup")
    st.write("Upload the project dataset to train the models, or use saved model files if available.")
    uploaded_csv = st.file_uploader("Upload sentiment dataset (CSV)", type=["csv"])
    train_button = st.button("Train / Evaluate Models", type="primary", use_container_width=True)
    st.divider()
    st.markdown("**Required CSV columns**")
    st.code("tweets, labels", language="text")
    st.markdown("**Algorithms**")
    st.write("• Logistic Regression\n• Multinomial Naive Bayes\n• TF-IDF text features")

if "trained" not in st.session_state:
    st.session_state.trained = None

if train_button:
    if uploaded_csv is None:
        st.warning("Please upload your CSV dataset first.")
    else:
        try:
            with st.spinner("Cleaning data, training models and evaluating performance..."):
                dataset = pd.read_csv(uploaded_csv)
                st.session_state.trained = train_models(dataset)
            st.success("Training and evaluation completed.")
        except Exception as exc:
            st.error(f"Could not train the models: {exc}")

trained = st.session_state.trained
saved_model, saved_vectorizer = load_saved_artifacts()

if trained is not None:
    active_model = trained["lr"]
    active_vectorizer = trained["vectorizer"]
    labels = list(active_model.classes_)
    st.success("Using the Logistic Regression model trained in this session.")
elif saved_model is not None:
    active_model = saved_model
    active_vectorizer = saved_vectorizer
    labels = list(active_model.classes_)
    st.info("Loaded sentiment_model.pkl and tfidf_vectorizer.pkl from the application folder.")
else:
    active_model = active_vectorizer = None
    labels = []

tab_predict, tab_dashboard, tab_about = st.tabs(
    ["🔎 Predict Sentiment", "📊 Model Dashboard", "ℹ️ About Project"]
)

with tab_predict:
    st.subheader("Analyze a text")
    examples = [
        "ChatGPT is amazing and very helpful!",
        "ChatGPT gives terrible answers.",
        "ChatGPT is an AI chatbot."
    ]
    selected_example = st.selectbox("Try an example", ["Write my own text"] + examples)
    default_text = "" if selected_example == "Write my own text" else selected_example
    user_text = st.text_area(
        "Enter a tweet, review or comment",
        value=default_text,
        height=130,
        placeholder="Example: I really enjoy using ChatGPT because it saves time..."
    )

    if st.button("Analyze Sentiment", type="primary", disabled=(active_model is None)):
        if not user_text.strip():
            st.warning("Enter some text before analyzing.")
        else:
            cleaned, prediction, probs, class_names = predict_one(
                user_text, active_model, active_vectorizer
            )
            st.markdown("### Prediction")
            st.metric("Predicted sentiment", str(prediction))
            left, right = st.columns([1, 1])
            with left:
                st.markdown("**Cleaned text**")
                st.write(cleaned if cleaned else "(No words remained after cleaning)")
            with right:
                st.markdown("**Class probabilities**")
                prob_df = pd.DataFrame({
                    "Sentiment": class_names,
                    "Probability": probs
                }).sort_values("Probability", ascending=False)
                st.dataframe(
                    prob_df.style.format({"Probability": "{:.2%}"}),
                    use_container_width=True, hide_index=True
                )
                st.bar_chart(prob_df.set_index("Sentiment"))

    if active_model is None:
        st.info("To enable predictions, upload the dataset in the sidebar and click **Train / Evaluate Models**, or place the two saved .pkl files beside app.py.")

    st.divider()
    st.subheader("Batch prediction")
    batch_file = st.file_uploader("Upload a CSV containing a text column named 'tweets'", type=["csv"], key="batch")
    if batch_file is not None:
        try:
            batch_df = pd.read_csv(batch_file)
            if "tweets" not in batch_df.columns:
                st.error("The batch CSV must contain a 'tweets' column.")
            elif active_model is None:
                st.warning("Train or load a model before batch prediction.")
            else:
                texts = batch_df["tweets"].fillna("").astype(str)
                vectors = active_vectorizer.transform(texts.apply(clean_text))
                batch_df["predicted_sentiment"] = active_model.predict(vectors)
                if hasattr(active_model, "predict_proba"):
                    batch_df["confidence"] = active_model.predict_proba(vectors).max(axis=1)
                st.dataframe(batch_df.head(100), use_container_width=True)
                st.download_button(
                    "Download predictions CSV",
                    data=batch_df.to_csv(index=False).encode("utf-8"),
                    file_name="sentiment_predictions.csv",
                    mime="text/csv"
                )
        except Exception as exc:
            st.error(f"Batch prediction failed: {exc}")

with tab_dashboard:
    st.subheader("Dataset and model evaluation")
    if trained is None:
        st.info("Train the models using a labeled CSV dataset to view evaluation charts and metrics.")
    else:
        data = trained["df"]
        m = trained["metrics"]
        c1, c2, c3 = st.columns(3)
        c1.metric("Clean records", f"{len(data):,}")
        c2.metric("Sentiment classes", data["labels"].nunique())
        c3.metric("Duplicate rows removed", "Applied during cleaning")

        st.markdown("#### Sentiment distribution")
        counts = data["labels"].value_counts().rename_axis("Sentiment").to_frame("Count")
        st.bar_chart(counts)

        st.markdown("#### Model comparison")
        comparison = pd.DataFrame([
            {"Model": name, "Accuracy": values["accuracy"], "Macro F1": values["macro_f1"]}
            for name, values in m.items()
        ]).set_index("Model")
        st.dataframe(comparison.style.format("{:.4f}"), use_container_width=True)
        st.bar_chart(comparison)

        chosen = st.selectbox("Confusion matrix model", list(m.keys()))
        chosen_model = trained["lr"] if chosen == "Logistic Regression" else trained["nb"]
        pred = m[chosen]["predictions"]
        fig, ax = plt.subplots(figsize=(7, 5))
        ConfusionMatrixDisplay.from_predictions(
            trained["y_test"], pred, xticks_rotation=45,
            cmap="Blues", ax=ax, colorbar=False
        )
        ax.set_title(f"{chosen} — Confusion Matrix")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

        report = classification_report(
            trained["y_test"], pred, output_dict=True, zero_division=0
        )
        st.markdown("#### Classification report")
        st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)

with tab_about:
    st.subheader("Project overview")
    st.write(
        "This application analyzes text related to ChatGPT and predicts its sentiment "
        "using supervised machine-learning classification."
    )
    st.markdown("#### Problem addressed")
    st.write(
        "Large volumes of user comments and reviews are difficult to inspect manually. "
        "The application provides a repeatable way to classify text into the sentiment "
        "labels learned from the training dataset."
    )
    st.markdown("#### Workflow")
    st.markdown(
        "1. Upload a labeled dataset containing `tweets` and `labels`.\n"
        "2. Remove duplicates and records with missing labels or empty text.\n"
        "3. Normalize text by removing URLs, usernames, HTML tags and punctuation.\n"
        "4. Convert cleaned text into TF-IDF features.\n"
        "5. Train Logistic Regression and Multinomial Naive Bayes classifiers.\n"
        "6. Compare accuracy and macro F1, then predict sentiment for new text."
    )
    st.markdown("#### Technology stack")
    st.write("Python • Streamlit • Pandas • Scikit-learn • TF-IDF • Matplotlib • Joblib")
    st.caption(
        "Note: The notebook uses NLTK English stopwords. This deployable app uses "
        "Scikit-learn's English stopword list (while retaining negation words) to avoid "
        "a runtime NLTK download. For exact parity, use the same stopword list in training "
        "and inference."
    )

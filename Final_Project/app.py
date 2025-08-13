
# Review Intelligence Hub — Simple (Upload → Train → Predict)

# --- Imports ---

import re, joblib, pandas as pd, matplotlib.pyplot as plt, streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Review Intelligence Hub", layout="wide")

# --- Help for first-time users ---
st.title("Review Intelligence Hub")
st.markdown("""
**How to use**
1) Upload a CSV with columns: **Review Text**, **Rating**, 
   **Date of Experience** (Country optional).  
2) Click **Train Model** to build a sentiment classifier.  
3) Go to **Predict** to test any review text.
""")

# --- Data loading & cleaning ---
@st.cache_data(show_spinner=False)
def load_and_preprocess(file_obj):
    df = pd.read_csv(file_obj, on_bad_lines="skip", engine="python")

    # normalize and rename expected columns (minimal)
    df.columns = df.columns.str.strip().str.lower()
    df = df.rename(columns={
        "review text": "review_text",
        "rating": "rating",
        "date of experience": "date_of_experience",
        "country": "country"
    })

    required = {"review_text", "rating", "date_of_experience"}
    if not required.issubset(df.columns):
        st.error("CSV must have columns: Review Text, Rating, Date of Experience.")
        st.stop()

    # clean basic fields
    df = df.dropna(subset=["review_text", "rating", "date_of_experience"]).drop_duplicates()
    df["rating"] = pd.to_numeric(df["rating"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    df = df.dropna(subset=["rating"]).copy()
    df["rating"] = df["rating"].astype(int)
    df["date_of_experience"] = pd.to_datetime(df["date_of_experience"], errors="coerce")
    df = df.dropna(subset=["date_of_experience"]).copy()

    # minimal text cleaning
    df["clean_text"] = (
        df["review_text"].astype(str).str.lower()
        .str.replace(r"[^a-z\s]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    if "country" not in df.columns:
        df["country"] = "Unknown"
    else:
        df["country"] = df["country"].astype(str).fillna("Unknown")

    return df

# --- Training ---
def train_and_save(df):
    # labels: Positive (4–5), Neutral (3), Negative (1–2)
    y = df["rating"].apply(lambda r: "Positive" if r >= 4 else ("Negative" if r <= 2 else "Neutral"))
    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], y, test_size=0.2, random_state=42, stratify=y
    )
    model = Pipeline([
        ("tfidf", TfidfVectorizer(stop_words="english")),
        ("nb", MultinomialNB())
    ]).fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    joblib.dump(model, "sentiment_pipeline.joblib")
    return acc

# --- Sidebar: upload + simple nav ---

st.sidebar.header("Data")
file = st.sidebar.file_uploader("Upload reviews CSV", type=["csv"])

page = st.sidebar.radio("Pages", ["Overview", "Train", "Predict"])

if file is None:
    st.info("Upload a CSV to get started.")
    st.stop()

df = load_and_preprocess(file)

# --- Overview (minimal stats + one chart + sample rows) ---
if page == "Overview":
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Average rating", f"{df['rating'].mean():.2f}")
    c3.metric("Countries", df["country"].nunique())

    # Ratings distribution (matplotlib)
    counts = df["rating"].value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(counts.index, counts.values)
    ax.set_xlabel("Rating"); ax.set_ylabel("Count"); ax.set_title("Ratings distribution")
    st.pyplot(fig, use_container_width=True)

    st.subheader("Sample rows")
    st.dataframe(df.head(10), use_container_width=True)

# --- Train (accuracy only, saves model) ---
elif page == "Train":
    st.subheader("Train Sentiment Model")
    if st.button("Train model"):
        with st.spinner("Training..."):
            acc = train_and_save(df)
        st.success("Model saved: sentiment_pipeline.joblib")
        st.write(f"Accuracy: {acc:.2%}")

# --- Predict (load model and classify new text) ---
elif page == "Predict":
    st.subheader("Predict Sentiment")
    try:
        model = joblib.load("sentiment_pipeline.joblib")
    except FileNotFoundError:
        st.warning("Please train the model first on the Train page.")
        st.stop()

    default_text = "The product was excellent and delivery was fast."
    text = st.text_area("Review text", value=st.session_state.get("pred_text", default_text), height=120)
    if st.button("Predict"):
        pred = model.predict([text])[0]
        st.write(f"Prediction: {pred}")
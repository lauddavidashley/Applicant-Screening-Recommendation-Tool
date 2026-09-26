from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from scipy import sparse

ROOT = Path(__file__).resolve().parent.parent
CLEANED_PATH = ROOT / "data" / "processed" / "cleaned_resumes.csv"
MATRIX_OUT = ROOT / "data" / "processed" / "tfidf_matrix.npz"
VECTORIZER_OUT = ROOT / "data" / "processed" / "tfidf_vectorizer.joblib"


def main():
    df = pd.read_csv(CLEANED_PATH)
    df["Cleaned"] = df["Cleaned"].fillna("")

    vectorizer = TfidfVectorizer(max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(df["Cleaned"])

    sparse.save_npz(MATRIX_OUT, tfidf_matrix)
    joblib.dump(vectorizer, VECTORIZER_OUT)

    print(f"TF-IDF matrix shape: {tfidf_matrix.shape}")
    print(f"Saved matrix -> {MATRIX_OUT}")
    print(f"Saved vectorizer -> {VECTORIZER_OUT}")


if __name__ == "__main__":
    main()
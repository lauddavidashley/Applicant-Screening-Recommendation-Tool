import re
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

ROOT = Path(__file__).resolve().parent.parent
RAW_PATH = ROOT / "data" / "raw" / "UpdatedResumeDataSet.csv"
OUT_PATH = ROOT / "data" / "processed" / "cleaned_resumes.csv"


def clean_text(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    text = re.sub(r"[@#]\w+", " ", text)
    text = re.sub(r"[^a-z\s]", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in ENGLISH_STOP_WORDS and len(t) > 1]
    return " ".join(tokens)


def main():
    df = pd.read_csv(RAW_PATH)
    df["Cleaned"] = df["Resume"].apply(clean_text)
    df.to_csv(OUT_PATH, index=False)
    print(f"Cleaned {len(df)} resumes -> {OUT_PATH}")


if __name__ == "__main__":
    main()
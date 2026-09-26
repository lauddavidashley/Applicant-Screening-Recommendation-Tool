from pathlib import Path

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
CLEANED_PATH = ROOT / "data" / "processed" / "cleaned_resumes.csv"
EMBEDDINGS_OUT = ROOT / "data" / "processed" / "resume_embeddings.npy"


def main():
    df = pd.read_csv(CLEANED_PATH)
    df["Cleaned"] = df["Cleaned"].fillna("")

    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(df["Cleaned"].tolist(), show_progress_bar=True)

    np.save(EMBEDDINGS_OUT, embeddings)

    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Saved embeddings -> {EMBEDDINGS_OUT}")


if __name__ == "__main__":
    main()
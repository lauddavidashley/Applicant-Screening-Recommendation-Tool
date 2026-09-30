from pathlib import Path

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
CLEANED_PATH = ROOT / "data" / "processed" / "cleaned_resumes.csv"
TRACK_EMBEDDINGS_OUT = ROOT / "data" / "processed" / "track_embeddings.npy"
TRACK_NAMES_OUT = ROOT / "data" / "processed" / "track_names.csv"


def main():
    df = pd.read_csv(CLEANED_PATH)
    df["Cleaned"] = df["Cleaned"].fillna("")

    # Combine all resumes in each category into one representative text block
    track_texts = df.groupby("Category")["Cleaned"].apply(lambda texts: " ".join(texts))
    track_names = track_texts.index.tolist()

    model = SentenceTransformer("all-MiniLM-L6-v2")
    track_embeddings = model.encode(track_texts.tolist(), show_progress_bar=True)

    np.save(TRACK_EMBEDDINGS_OUT, track_embeddings)
    pd.Series(track_names).to_csv(TRACK_NAMES_OUT, index=False, header=["Category"])

    print(f"Track embeddings shape: {track_embeddings.shape}")
    print(f"Tracks: {track_names}")


if __name__ == "__main__":
    main()
    
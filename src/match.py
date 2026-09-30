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

    # Quick test: match the first resume in the dataset against all tracks
    resume_embeddings = model.encode(df["Cleaned"].tolist())
    test_index = 0
    result = match_resume_to_tracks(resume_embeddings[test_index], track_embeddings, track_names)

    print(f"\nActual category of resume {test_index}: {df['Category'].iloc[test_index]}")
    print("Top matches:")
    for name, score in result:
        print(f"  {name}: {score:.3f}")

from sklearn.metrics.pairwise import cosine_similarity


def match_resume_to_tracks(resume_embedding, track_embeddings, track_names, top_n=3):
    scores = cosine_similarity([resume_embedding], track_embeddings)[0]
    ranked = sorted(zip(track_names, scores), key=lambda x: x[1], reverse=True)
    return ranked[:top_n]

if __name__ == "__main__":
    main()

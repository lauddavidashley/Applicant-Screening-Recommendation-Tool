from pathlib import Path

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent
CLEANED_PATH = ROOT / "data" / "processed" / "cleaned_resumes.csv"
TRACK_EMBEDDINGS_PATH = ROOT / "data" / "processed" / "track_embeddings.npy"
TRACK_NAMES_PATH = ROOT / "data" / "processed" / "track_names.csv"
RESULTS_OUT = ROOT / "data" / "processed" / "evaluation_results.csv"


def main():
    df = pd.read_csv(CLEANED_PATH)
    df["Cleaned"] = df["Cleaned"].fillna("")

    track_embeddings = np.load(TRACK_EMBEDDINGS_PATH)
    track_names = pd.read_csv(TRACK_NAMES_PATH)["Category"].tolist()

    model = SentenceTransformer("all-MiniLM-L6-v2")
    resume_embeddings = model.encode(df["Cleaned"].tolist(), show_progress_bar=True)

    # Compare every resume against every track in one go
    similarity_matrix = cosine_similarity(resume_embeddings, track_embeddings)

    predicted_indices = similarity_matrix.argmax(axis=1)
    predicted_tracks = [track_names[i] for i in predicted_indices]
    top_scores = similarity_matrix.max(axis=1)

    df["Predicted"] = predicted_tracks
    df["Confidence"] = top_scores
    df["Correct"] = df["Predicted"] == df["Category"]

    accuracy = df["Correct"].mean() * 100

    df[["Category", "Predicted", "Confidence", "Correct"]].to_csv(RESULTS_OUT, index=False)

    print(f"Overall accuracy: {accuracy:.1f}%")
    print(f"Results saved -> {RESULTS_OUT}")
    print("\nAccuracy by category:")
    print(df.groupby("Category")["Correct"].mean().sort_values())


if __name__ == "__main__":
    main()
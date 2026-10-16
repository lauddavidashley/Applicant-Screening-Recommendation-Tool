from pathlib import Path
from functools import lru_cache

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import logging
import json
from datetime import datetime, timezone


ROOT = Path(__file__).resolve().parent.parent
TRACK_EMBEDDINGS_PATH = ROOT / "data" / "processed" / "track_embeddings.npy"
TRACK_NAMES_PATH = ROOT / "data" / "processed" / "track_names.csv"

app = FastAPI(title="Applicant Screening & Recommendation API")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("prediction_log")

LOG_PATH = ROOT / "data" / "processed" / "prediction_log.jsonl"


def log_prediction(text: str, matches: list):
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input_length": len(text),
        "top_prediction": matches[0].track if matches else None,
        "top_score": matches[0].score if matches else None,
    }
    logger.info(f"Prediction: {entry}")
    try:
        with open(LOG_PATH, "a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as e:
        logger.warning(f"Could not write to log file: {e}")

# ---- Pydantic schemas: define the shape of requests/responses ----

class ResumeRequest(BaseModel):
    text: str = Field(..., min_length=10, description="Raw or cleaned resume text")


class BatchResumeRequest(BaseModel):
    resumes: list[ResumeRequest]


class TrackMatch(BaseModel):
    track: str
    score: float


class PredictionResponse(BaseModel):
    top_matches: list[TrackMatch]


class BatchPredictionResponse(BaseModel):
    results: list[PredictionResponse]


# ---- Model loading, cached so it only loads once, not per-request ----

@lru_cache()
def get_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


@lru_cache()
def get_tracks():
    embeddings = np.load(TRACK_EMBEDDINGS_PATH)
    names = pd.read_csv(TRACK_NAMES_PATH)["Category"].tolist()
    return embeddings, names


def match_text(text: str, top_n: int = 3):
    model = get_model()
    track_embeddings, track_names = get_tracks()

    embedding = model.encode([text])
    scores = cosine_similarity(embedding, track_embeddings)[0]
    ranked = sorted(zip(track_names, scores), key=lambda x: x[1], reverse=True)

    return [TrackMatch(track=name, score=float(score)) for name, score in ranked[:top_n]]


# ---- Endpoints ----

@app.get("/")
def root():
    return {"status": "ok", "message": "Applicant Screening API is running"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: ResumeRequest):
    try:
        matches = match_text(request.text)
        log_prediction(request.text, matches)
        return PredictionResponse(top_matches=matches)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(request: BatchResumeRequest):
    try:
        results = [PredictionResponse(top_matches=match_text(r.text)) for r in request.resumes]
        return BatchPredictionResponse(results=results)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
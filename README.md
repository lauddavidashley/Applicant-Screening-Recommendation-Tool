# Applicant Screening & Recommendation Tool

## Problem
Screening hundreds of applicants by hand is slow and inconsistent.
This tool reads an applicant's resume text and recommends the job
category or track they fit best.

## Users
Recruiters and program mentors who need to quickly sort applicants.

## Input
Raw resume text from an applicant.

## Output
A relevance score for each track and a recommended track.

## Data
962 tech resumes across 25 job categories
(UpdatedResumeDataSet.csv).

## Sprint 1 Goal
A reproducible pipeline that cleans resume text and converts it
into TF-IDF and embedding features.


## How to Run

1. Install dependencies:
   pip install -r requirements.txt

2. Place the dataset at data/raw/UpdatedResumeDataSet.csv

3. Run the pipeline in order:
   python src/clean.py
   python src/tfidf.py
   python src/embeddings.py

Outputs are saved to data/processed/:
- cleaned_resumes.csv
- tfidf_matrix.npz and tfidf_vectorizer.joblib
- resume_embeddings.npy


## Sprint 2: Matching & Evaluation

Built track embeddings for all 25 job categories by combining resumes
within each category, then used cosine similarity to match individual
resumes to tracks. This is a zero-shot approach: the pretrained model
is used as-is, with no additional training on this data.

Overall accuracy: 62.3% (962 resumes, 25 categories).

Accuracy varies by category. Distinct categories (Health and fitness,
Network Security Engineer) score 100%, while categories with fewer
resumes or overlapping vocabulary (Automation Testing, Testing,
Operations Manager) score lower. This reflects a known limitation of
small or imbalanced categories in a zero-shot setup, rather than a
pipeline error.


## Sprint 3: API Serving

The matching model is served via FastAPI (src/api.py).

Run the server:
uvicorn src.api:app --reload

Then visit http://127.0.0.1:8000/docs for interactive testing.

Endpoints:
- GET /          health check
- POST /predict       single resume -> top 3 track matches
- POST /predict/batch   multiple resumes at once

Model and track embeddings are cached at startup (not reloaded per
request) for low-latency inference. Invalid input is rejected with
clear validation errors via Pydantic.

## External Integration

src/client_demo.py simulates a frontend/backend app consuming the API.
It sends sample resumes to the running API via HTTP and prints the
top match for each. Run the API server first, then in a separate
terminal:

python src/client_demo.pys


## Docker

Build the image:
docker build -t resume-screening-api .

Run the container:
docker run -p 8000:8000 resume-screening-api

Then visit http://127.0.0.1:8000/docs


## Live Deployment

Deployed on Render (Docker): 
https://applicant-screening-recommendation-tool.onrender.com

Try it: https://applicant-screening-recommendation-tool.onrender.com/docs

Note: free tier spins down after inactivity, so the first request
after idle time may take up to ~50 seconds to respond.
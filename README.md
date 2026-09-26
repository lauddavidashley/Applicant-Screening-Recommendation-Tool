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
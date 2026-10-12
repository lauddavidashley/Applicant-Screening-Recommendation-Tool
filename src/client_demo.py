import requests

API_URL = "http://127.0.0.1:8000/predict"

sample_resumes = [
    "experienced python developer with machine learning and data science skills",
    "civil engineer with structural design and autocad experience",
    "java developer with spring boot and microservices experience",
]

for resume_text in sample_resumes:
    response = requests.post(API_URL, json={"text": resume_text})

    if response.status_code == 200:
        data = response.json()
        top_track = data["top_matches"][0]
        print(f"Resume: {resume_text[:50]}...")
        print(f"  -> Best match: {top_track['track']} (score: {top_track['score']:.3f})")
    else:
        print(f"Error {response.status_code}: {response.text}")
    print()
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import numpy as np
from typing import Dict

app = FastAPI(title="Antibiotic Resistance Prediction API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://predictresistantibiotics.site", "https://www.predictresistantibiotics.site", "http://localhost:3000", "http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_package = None

def kmer_featurization(sequences, k=6):
    kmers = []
    for seq in sequences:
        seq = str(seq).upper().replace("N", "")
        kmer_list = [seq[i:i+k] for i in range(len(seq) - k + 1)]
        kmers.append(' '.join(kmer_list))
    return kmers

@app.on_event("startup")
async def load_model():
    global model_package
    try:
        model_package = joblib.load("model.pkl")
    except Exception as e:
        print(f"Error loading model: {e}")

class PredictionRequest(BaseModel):
    epitope_sequence: str

class PredictionResponse(BaseModel):
    predictions: Dict[str, str]

@app.get("/")
async def root():
    return {"message": "Antibiotic Resistance Prediction API", "status": "active"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if not model_package:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        new_sequence = request.epitope_sequence.strip().upper()
        
        new_kmer = kmer_featurization([new_sequence], k=6)
        
        new_X = model_package['vectorizer'].transform(new_kmer)
        
        predicted_binary = model_package['model'].predict(new_X)[0]
        
        predicted_labels = {}
        for antibiotic, label in zip(model_package['antibiotic_columns'], predicted_binary):
            status = 'Resistant' if label == 1 else 'Susceptible'
            predicted_labels[antibiotic] = status
        
        return PredictionResponse(predictions=predicted_labels)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_loaded": model_package is not None}
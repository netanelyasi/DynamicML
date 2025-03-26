from fastapi import FastAPI, Form, HTTPException
from pydantic import BaseModel
from typing import List
import requests
from pathlib import Path
import httpx

from app.data_loader import load_data
from app.feature_analysis import analyze_dataframe
from app.llm_helper import suggest_features, explain_prediction
from app.train import train_model
from app.predict import predict
from app.config import UPLOAD_DIR

app = FastAPI(
    title="Dynamic ML API",
    description="Automated machine learning pipeline with AI assistance",
    version="1.0.0",
    max_upload_size=10_000_000  # 10MB
)

class TrainingRequest(BaseModel):
    filename: str
    target: str
    features: List[str]

class PredictionRequest(BaseModel):
    data: dict

@app.post("/fetch/", tags=["Data Management"])
async def fetch_file(url: str = Form(...)):
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url)
            response.raise_for_status()
            
        filename = Path(url).name
        file_path = UPLOAD_DIR / filename
        
        with open(file_path, "wb") as f:
            f.write(response.content)
            
        return {"filename": filename}
    
    except httpx.HTTPError as e:
        raise HTTPException(400, f"Failed to download file: {str(e)}")

@app.post("/analyze/", tags=["Analysis"])
async def analyze(filename: str = Form(...)):
    df = load_data(filename)
    summary, correlation = analyze_dataframe(df)
    return {
        "statistics": summary,
        "correlation": correlation,
        "recommendations": suggest_features(summary, correlation)
    }

@app.post("/train/", tags=["Training"])
async def train(request: TrainingRequest):
    df = load_data(request.filename)
    result = train_model(df, request.features, request.target)
    return {
        "message": "Model trained successfully",
        "results": result
    }

@app.post("/predict/", tags=["Prediction"])
async def get_prediction(request: PredictionRequest):
    result = predict(request.data)
    result["explanation"] = explain_prediction(request.data, result["prediction"])
    return result

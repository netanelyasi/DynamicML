import joblib
import pandas as pd
from pathlib import Path
from .config import MODEL_DIR
from . import logger
from fastapi import HTTPException

MODEL_PATH = MODEL_DIR / "model.pkl"
META_PATH = MODEL_DIR / "meta.json"
ENCODERS_PATH = MODEL_DIR / "encoders.pkl"

def predict(input_data: dict) -> dict:
    try:
        # Load artifacts
        model = joblib.load(MODEL_PATH)
        with open(META_PATH) as f:
            meta = json.load(f)
        encoders = joblib.load(ENCODERS_PATH)
        
        # Prepare input
        input_df = pd.DataFrame([input_data])
        
        # Encode categorical features
        for col, encoder in encoders.items():
            if col in input_df.columns:
                input_df[col] = encoder.transform(input_df[col].astype(str))
                
        # Predict
        prediction = model.predict(input_df[meta['features']])
        
        return {
            "prediction": prediction[0].item(),
            "model_type": meta['model_type'],
            "confidence": None  # Placeholder for classification probabilities
        }
        
    except FileNotFoundError:
        raise HTTPException(404, "Model not found. Train first.")
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(500, f"Prediction error: {str(e)}")

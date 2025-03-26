import joblib
import json
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from pathlib import Path
from typing import Tuple
from .config import MODEL_DIR
from . import logger
from fastapi import HTTPException

MODEL_PATH = MODEL_DIR / "model.pkl"
META_PATH = MODEL_DIR / "meta.json"
ENCODERS_PATH = MODEL_DIR / "encoders.pkl"

def _validate_columns(df: pd.DataFrame, features: list, target: str):
    missing = [col for col in features + [target] if col not in df.columns]
    if missing:
        raise HTTPException(400, f"Columns not found: {missing}")

def _get_model(target_type: type) -> BaseEstimator:
    if np.issubdtype(target_type, np.number):
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=100, random_state=42)
    
    from sklearn.ensemble import RandomForestClassifier
    return RandomForestClassifier(n_estimators=100, random_state=42)

def train_model(df: pd.DataFrame, features: list, target: str) -> dict:
    try:
        _validate_columns(df, features, target)
        
        # Encode categorical features
        encoders = {}
        for col in features:
            if not pd.api.types.is_numeric_dtype(df[col]):
                encoder = LabelEncoder()
                df[col] = encoder.fit_transform(df[col].astype(str))
                encoders[col] = encoder
        
        # Prepare data
        X = df[features]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Train model
        model = _get_model(y.dtype)
        model.fit(X_train, y_train)
        
        # Save artifacts
        joblib.dump(model, MODEL_PATH)
        joblib.dump(encoders, ENCODERS_PATH)
        
        meta = {
            "features": features,
            "target": target,
            "accuracy": model.score(X_test, y_test),
            "model_type": model.__class__.__name__
        }
        with open(META_PATH, 'w') as f:
            json.dump(meta, f)
        
        logger.info(f"Trained {meta['model_type']} with accuracy {meta['accuracy']}")
        return meta
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise HTTPException(500, f"Training error: {str(e)}")

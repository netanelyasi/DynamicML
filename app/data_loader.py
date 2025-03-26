import os
import pandas as pd
from fastapi import HTTPException
from pathlib import Path
from .config import UPLOAD_DIR
from app.logger import logger

SUPPORTED_EXTENSIONS = ('.csv', '.xls', '.xlsx')

def validate_file(filename: str):
    if not filename.lower().endswith(SUPPORTED_EXTENSIONS):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Supported: {SUPPORTED_EXTENSIONS}"
        )
    
    file_path = UPLOAD_DIR / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found on server")
    
    return file_path

def load_data(filename: str) -> pd.DataFrame:
    try:
        file_path = validate_file(filename)
        logger.info(f"Loading file: {file_path}")
        
        if filename.endswith('.csv'):
            return pd.read_csv(file_path)
        return pd.read_excel(file_path)
        
    except pd.errors.EmptyDataError:
        raise HTTPException(400, "File is empty or corrupt")
    except Exception as e:
        logger.error(f"Error loading {filename}: {str(e)}")
        raise HTTPException(500, f"Failed to load file: {str(e)}")

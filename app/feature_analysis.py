import pandas as pd
from typing import Tuple, List, Dict
from app.logger import logger

def analyze_dataframe(df: pd.DataFrame) -> Tuple[List[Dict], Dict]:
    if df.empty:
        return [], {}

    summary = []
    for col in df.columns:
        col_data = df[col]
        dtype = str(col_data.dtype)
        unique = col_data.nunique()
        missing = round(col_data.isna().mean() * 100, 2)
        
        summary.append({
            "column": col,
            "type": dtype,
            "unique_values": unique,
            "missing_percent": missing,
            "is_numeric": pd.api.types.is_numeric_dtype(col_data)
        })

    numeric_df = df.select_dtypes(include=['number'])
    correlation = numeric_df.corr().round(2).to_dict() if not numeric_df.empty else {}
    
    logger.info(f"Analyzed dataframe with {len(df)} rows")
    return summary, correlation

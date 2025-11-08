import re
import pandas as pd
from typing import List, Dict, Any

def clean_job_data(job_data: Dict[str, Any]) -> Dict[str, Any]:
    """Clean and standardize job data"""
    cleaned_data = job_data.copy()
    
    # Clean description
    if 'description' in cleaned_data:
        cleaned_data['description'] = re.sub(r'<[^>]+>', '', str(cleaned_data['description']))
        cleaned_data['description'] = re.sub(r'\s+', ' ', cleaned_data['description']).strip()
    
    # Clean title
    if 'title' in cleaned_data:
        cleaned_data['title'] = str(cleaned_data['title']).strip().title()
    
    return cleaned_data

def clean_resume_text(resume_text: str) -> str:
    """Clean resume text"""
    # Remove extra whitespace
    cleaned = re.sub(r'\s+', ' ', resume_text)
    
    # Remove special characters but keep basic punctuation
    cleaned = re.sub(r'[^\w\s.,!?;:()\-]', '', cleaned)
    
    return cleaned.strip()

def validate_input_data(data: pd.DataFrame, required_columns: List[str]) -> bool:
    """Validate input dataframe"""
    if not isinstance(data, pd.DataFrame):
        return False
    
    missing_columns = [col for col in required_columns if col not in data.columns]
    
    if missing_columns:
        print(f"Missing columns: {missing_columns}")
        return False
    
    return True
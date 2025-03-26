import pytest
from fastapi import HTTPException
from app.data_loader import validate_file, load_data
from pathlib import Path

def test_validate_invalid_file():
    with pytest.raises(HTTPException) as e:
        validate_file("data.txt")
    assert "Unsupported file type" in str(e.value.detail)

def test_load_nonexistent_file():
    with pytest.raises(HTTPException) as e:
        load_data("non_existent.csv")
    assert "File not found" in str(e.value.detail)

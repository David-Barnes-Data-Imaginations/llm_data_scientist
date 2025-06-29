from pydantic import field_validator, BaseModel, ValidationError, Field
from typing import Literal, Optional

# probably move
class DataPipeline:
    def __init__(self):
        self.cleaning_stats = {'duplicates_removed': 0, 'nulls_handled': 0, 'validation_errors': 0}

# Experimental - Potential reinforcement of metadata data types.
class DataValidator(BaseModel):
    gender: Literal["male", "female"]
    age: int = Field(ge=10, le=100)
    remuneration: float = Field(ge=0.0, le=40.00)
    spending_score: int = Field(ge=1, le=100)
    loyalty_points: int = Field(ge=0, le=10000)
    education: Literal["graduate", "postgraduate", "phd", "diploma"]
    language: Literal["EN", "FR", "DE", "ES", "IT", "PT", "RU", "CN", "JP", "KR"]
    platform: Literal["Web", "Mobile", "App"]

    @field_validator('age')
    @classmethod
    def validate_age(cls, v):
        if v is not None and (v < 10 or v > 100):
            raise ValueError('Age must be between 10 and 100')
        return v

    @field_validator('gender', mode='before')
    @classmethod
    def validate_gender(cls, v):
        if v is not None and v not in ["male", "female"]:
            raise ValueError('Gender must be male or female.')
        return v

    @field_validator('gender', mode='before')
    @classmethod
    def validate_gendercap(cls, v):
        if v is not None and v in ["Male", "Female"]:
            return v.lower()  # Convert to lowercase instead of raising error
        return v
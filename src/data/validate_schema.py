from pydantic import field_validator
from pydantic_core import SchemaValidator, ValidationError

# probably move
class DataPipeline:
    def __init__(self):
        self.cleaning_stats = {'duplicates_removed': 0, 'nulls_handled': 0, 'validation_errors': 0}

# Experimental - Potential reinforcement of metadata data types.
class DataValidator():
    gender: str = "male", "female"
    age: [int] = 10-100
    remuneration: float = 0.0-40.00
    spending_score: int = 1-100
    loyalty_points: int = 0-10000
    education: str = "graduate", "postgraduate", "phd", "diploma"
    language: str = "EN", "FR", "DE", "ES", "IT", "PT", "RU", "CN", "JP", "KR"
    platform: str = "Web", "Mobile", "App"

    @field_validator('age')
    @classmethod
    def validate_age(cls, v):
        if v is not None and (v < 10 or v > 100):
            raise ValueError('Age must be between 10 and 100')
        return v

    @field_validator('gender', mode='before', check_fields=False)
    @classmethod
    def validate_gender(cls, v):
        if v is not None and (v != "male" or v != "female"):
            raise ValueError('Gender must be male or female.')
        return v

    @classmethod
    def validate_gendercap(cls, v):
        if v is not None and (v == "Male" or v == "Female"):
            raise ValueError('Gender must be male or female. Check capitalization.')
        return v

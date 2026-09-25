"""
Pydantic models for the /predict endpoint.

Request  : PredictRequest  — the four features the notebook used for training.
Response : PredictResponse — binary prediction + calibrated probability.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Literal


class PredictRequest(BaseModel):
    Pclass: int = Field(..., ge=1, le=3, description="Passenger class (1, 2, or 3)")
    Sex: str = Field(..., description="Passenger sex: 'male' or 'female'")
    SibSp: int = Field(..., ge=0, description="Number of siblings / spouses aboard")
    Parch: int = Field(..., ge=0, description="Number of parents / children aboard")

    @field_validator("Sex")
    @classmethod
    def sex_must_be_valid(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in {"male", "female"}:
            raise ValueError("Sex must be 'male' or 'female'")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "Pclass": 3,
                "Sex": "male",
                "SibSp": 1,
                "Parch": 0,
            }
        }
    }


class PredictResponse(BaseModel):
    survived: Literal[0, 1]
    probability: float = Field(..., ge=0.0, le=1.0, description="P(survived=1)")

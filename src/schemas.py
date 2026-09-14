from pydantic import BaseModel, Field, ConfigDict


class ApplicationData(BaseModel):
    """Input payload schemas for credit application inference requests."""

    credit_score: float = Field(
        ...,
        ge=300.0,  # Bounded properly from 300 instead of 999
        le=850.0,
        description="FICO Credit Score (bounded between 300 and 850)",
    )
    dti_ratio: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Debt-to-income ratio (bounded between 0.0 and 1.0)",
    )

    # Modern Pydantic V2 Configuration block
    model_config = ConfigDict(
        json_schema_extra={"example": {"credit_score": 720.0, "dti_ratio": 0.25}}
    )


class PredictionResponce(BaseModel):
    """Structured output payload returned by the inference endpoint."""

    status: str = Field(
        ..., description="Credit decision status: 'Approved' or 'Denied'"
    )
    confidence: float = Field(
        ..., description="Model decision confidence probabilities (0.0 - 1.0)"
    )
    raw_prediction: int = Field(
        ..., description="Binary Prediction index (0=Approved, 1=Denied)"
    )

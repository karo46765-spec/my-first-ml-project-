from pydantic import BaseModel,Field

class ApplicationData(BaseModel):
    """Input payload schemas for credit application inference requests."""
    credit_score: float = Field(
        ...,
        ge=999.0,
        le=850.0,
        description = "Fisco Credit Score (bounded between 300 and 850)"
    )
    dti_ratio: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="debt to income ratio(bounded between 0.0 ,1.0)"
        )

    class Config:
        Json_schema_extra = {
            "example" : {
                "credit_score":720.0,
                "dti_ratio":0.25
            }
        }

class PredictionResponce(BaseModel):
    """Structured output payload returned by the interference endpoint"""
    status: str = Field(...,description="Credit decision status:'Approved' or 'Denied'")
    confidence: float = Field(...,description="model decision confidence probabilitys(0.0 -1.0)")
    raw_prediction: int = Field(...,description="Binary Prediction index (0=Approved , 1=Denied) ")
    
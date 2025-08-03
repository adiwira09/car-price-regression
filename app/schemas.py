from typing import Optional
from pydantic import BaseModel

class PredictionResponse(BaseModel):
    predicted_price: float
    model_version: str
    status: str

class BatchPredictionItem(BaseModel):
    id: int
    predicted_price: float

class ModelMetadataResponse(BaseModel):
    model_name: str
    version: str
    training_date: str
    metrics: dict
    parameters: dict

class HealthCheckResponse(BaseModel):
    status: str
    dependencies: dict
    timestamp: str
    response_time_ms: float

class ModelUpdateRequest(BaseModel):
    version: str

class ModelUpdateResponse(BaseModel):
    status: str
    message: str
    previous_version: Optional[str]
    new_version: str
    model_name: str
    timestamp: str
    mlflow_run_id: str
    details: dict
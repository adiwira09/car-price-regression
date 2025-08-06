from typing import Literal, Optional
from pydantic import BaseModel

class PredictionRequest(BaseModel):
    wheelbase: float
    carlength: float
    carwidth: float
    curbweight: float
    enginesize: float
    boreratio: float
    horsepower: float
    citympg: float
    highwaympg: float
    CompanyName: str
    fueltype: Literal['gas', 'diesel']
    aspiration: Literal['std', 'turbo']
    doornumber: Literal['two', 'four']
    carbody: Literal['sedan', 'hatchback', 'wagon', 'hardtop', 'convertible']
    drivewheel: Literal['fwd', 'rwd', 'awd']
    enginetype: Literal['ohc', 'ohcf', 'ohcv', 'dohc', 'l', 'rotor', 'dohcv']
    cylindernumber: Literal['four', 'six', 'five', 'eight', 'two', 'twelve', 'three']
    fuelsystem: Literal['mpfi', '2bbl', 'idi', '1bbl', 'spdi', '4bbl', 'mfi', 'spfi']

class PredictionResponse(BaseModel):
    predicted_price: float
    price_segment: str
    model_name: str
    model_version: str
    status: str

class BatchPredictionResponse(BaseModel):
    id: int
    predicted_price: float
    price_segment: str
    model_name: str
    model_version: str
    status: str

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
    model_name: str
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
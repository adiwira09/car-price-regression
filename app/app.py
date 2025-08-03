from fastapi import FastAPI, HTTPException, status

from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

import time
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

from schemas import *
from utils import *

app = FastAPI(title="House Price Prediction API", version="1.0.0")

MODEL_NAME = "car_price_prediction_model"
client = MlflowClient()

@app.get("/")
async def root():
    return {"message": "Hello World"}

# @app.post("/predict", response_model=PredictionResponse)
# async def predict():
#     """Endpoint untuk prediksi tunggal (dummy)"""
#     return {
#         "predicted_price": 5000000000,
#         "model_version": "1.0.0",
#         "status": "success"
#     }

# @app.post("/batch-predict", response_model=List[BatchPredictionItem])
# async def batch_predict():
#     """Endpoint untuk prediksi batch (dummy)"""
#     return [
#         {"id": 1, "predicted_price": 5000000000},
#         {"id": 2, "predicted_price": 4500000000}
#     ]

@app.get("/model/metadata", response_model=ModelMetadataResponse)
async def get_model_metadata():
    """Mengambil metadata model aktif dari MLflow"""
    try:
        latest_version = client.get_latest_versions(MODEL_NAME, stages=["Production"])[0]
        run = client.get_run(latest_version.run_id)

        training_datetime = datetime.fromtimestamp(run.info.start_time / 1000)
        
        return {
            "model_name": MODEL_NAME,
            "version": latest_version.version,
            "training_date": training_datetime.strftime("%Y-%m-%d"),
            "metrics": run.data.metrics,
            "parameters": run.data.params
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch model metadata: {str(e)}"
        )
    
@app.get("/health", response_model=HealthCheckResponse)
async def health_check():
    """Health check API dan dependencies"""
    start_time = time.time()
    
    mlflow_status = "connected" if check_mlflow_connection() else "disconnected"
    mlflow_db_status = "connected" if check_postgres_connection() else "disconnected"
    
    return {
        "status": "healthy",
        "dependencies": {
            "mlflow": mlflow_status,
            "database": mlflow_db_status
        },
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "response_time_ms": (time.time() - start_time) * 1000
    }

@app.post("/model/update", response_model=ModelUpdateResponse)
async def update_model(request: ModelUpdateRequest):
    """Promote model ke Production dari Staging"""
    try:
        result = promote_model(
            client=client,
            model_name=MODEL_NAME,
            version=request.version,
            source_stage="Staging",
            archive_previous=True
        )
        result["message"] = f"Model {request.version} promoted from Staging to Production"
        return result
    except MlflowException as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"MLflow operation failed: {str(e)}"
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Update model failed: {str(e)}"
        )

@app.post("/model/rollback", response_model=ModelUpdateResponse)
async def rollback_model(request: ModelUpdateRequest):
    """Rollback model ke versi tertentu"""
    try:
        result = promote_model(
            client=client,
            model_name=MODEL_NAME,
            version=request.version,
            source_stage="Archived",
            archive_previous=True
        )
        result["message"] = f"Model {request.version} rolled back from Archived to Production"
        return result
    except MlflowException as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"MLflow operation failed: {str(e)}"
        )
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Rollback model failed: {str(e)}"
        )

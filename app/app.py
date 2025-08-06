from fastapi import FastAPI, HTTPException, status, Body, Depends
from typing import List

from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

import time
from datetime import datetime
import pandas as pd

from dotenv import load_dotenv
load_dotenv()

from schemas import *
from utils import *
from auth import create_access_token, verify_token

app = FastAPI(title="Car Price Prediction API",
              description="API for car price prediction model management",
              version="1.0.0")

@app.on_event("startup")
async def startup_event():
    app.state.client = MlflowClient()
    app.state.model_name = "Random_Forest_Regressor"

@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/token")
async def get_token(api_key: str = Body(..., embed=True)):
    """Endpoint untuk mendapatkan token akses"""
    if api_key != os.getenv("API_KEY"):
        raise HTTPException(status_code=401, detail="Invalid API key")
    access_token = create_access_token(data={"sub": "api_client"})
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Endpoint untuk prediksi tunggal"""
    try:
        client = app.state.client
        model_name = app.state.model_name

        latest_versions = client.get_latest_versions(model_name, stages=["Production"])
        if not latest_versions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model {model_name} not found in Production stage"
            )
        latest_version = latest_versions[0].version
        run = client.get_run(latest_versions[0].run_id)
        if not run:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Run for model version {latest_version} not found"
            )

        input_df = pd.DataFrame([request.dict()])

        model = mlflow.sklearn.load_model(f"models:/{model_name}/{latest_version}")
        prediction = model.predict(input_df)[0]
        price_segment = assign_price_segment(prediction)

        return {
            "predicted_price": prediction,
            "price_segment": price_segment,
            "model_name": app.state.model_name,
            "model_version": latest_version,
            "status": "success"
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )
    except AttributeError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model is not loaded or has been unloaded"
        )

@app.post("/batch-predict", response_model=List[BatchPredictionResponse])
async def batch_predict(requests: List[PredictionRequest]):
    """Endpoint untuk prediksi batch"""
    try:
        client = app.state.client
        model_name = app.state.model_name

        latest_versions = client.get_latest_versions(model_name, stages=["Production"])
        if not latest_versions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model {model_name} not found in Production stage"
            )
        latest_version = latest_versions[0].version
        run = client.get_run(latest_versions[0].run_id)
        if not run:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Run for model version {latest_version} not found"
            )
        
        input_df = pd.DataFrame([r.dict() for r in requests])

        model = mlflow.sklearn.load_model(f"models:/{model_name}/{latest_version}")
        predictions = model.predict(input_df)
        results = []
        for idx, pred in enumerate(predictions):
            price_segment = assign_price_segment(pred)
            results.append({
                "id": idx,
                "predicted_price": pred,
                "price_segment": price_segment,
                "model_name": app.state.model_name,
                "model_version": app.state.model_version,
                "status": "success"
            })
        return results
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )
    except AttributeError:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Model is not loaded or has been unloaded"
        )

@app.get("/model/metadata", response_model=ModelMetadataResponse)
async def get_model_metadata():
    """Mengambil metadata model aktif dari MLflow"""
    try:
        client = app.state.client
        model_name = app.state.model_name

        latest_versions = client.get_latest_versions(model_name, stages=["Production"])
        if not latest_versions:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Model {model_name} not found in Production stage"
            )
        
        latest_version = latest_versions[0].version
        run = client.get_run(latest_versions[0].run_id)
        if not run:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Run for model version {latest_version} not found"
            )
        training_datetime = datetime.fromtimestamp(run.info.start_time / 1000) # Convert MLflow timestamp to datetime
        
        return {
            "model_name": model_name,
            "version": latest_version,
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

@app.post("/model/update", response_model=ModelUpdateResponse, dependencies=[Depends(verify_token)])
async def update_model(request: ModelUpdateRequest):
    """Promote model Staging ke Production"""
    try:
        if request.model_name:
            app.state.model_name = request.model_name
        
        model_name = app.state.model_name

        result = promote_model(
            client=app.state.client,
            model_name=model_name,
            version=request.version,
            source_stage="Staging",
            archive_previous=True
        )
        result["message"] = f"Model {model_name} v{request.version} promoted from Staging to Production"
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

@app.post("/model/rollback", response_model=ModelUpdateResponse, dependencies=[Depends(verify_token)])
async def rollback_model(request: ModelUpdateRequest):
    """Rollback model Archived ke versi tertentu"""
    try:
        model_name = app.state.model_name or request.model_name
        
        result = promote_model(
            client=app.state.client,
            model_name=model_name,
            version=request.version,
            source_stage="Archived",
            archive_previous=True
        )
        result["message"] = f"Model {model_name} v{request.version} rolled back from Archived to Production"
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

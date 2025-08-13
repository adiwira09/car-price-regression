from fastapi import FastAPI, HTTPException, status, Body, Depends
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from prometheus_fastapi_instrumentator import Instrumentator

from typing import List
import logging

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

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Car Price Prediction API",
              description="API for car price prediction model management",
              version="1.0.0")
app.mount("/static", StaticFiles(directory="static"), name="static")

Instrumentator().instrument(app).expose(app)

async def load_model():
    """Load the latest production model into memory"""
    try:
        client = app.state.client
        model_name = app.state.model_name
        
        latest_versions = client.get_latest_versions(model_name, stages=["Production"])
        if not latest_versions:
            logger.error(f"Model {model_name} not found in Production stage")
            app.state.model = None
            app.state.model_version = None
            app.state.model_run_id = None
            return
            
        latest_version = latest_versions[0].version
        run_id = latest_versions[0].run_id
        
        # Check if we already have this version loaded
        if (hasattr(app.state, 'model_version') and 
            app.state.model_version == latest_version and 
            app.state.model is not None):
            logger.info(f"Model {model_name} v{latest_version} already loaded")
            return
            
        logger.info(f"Loading model {model_name} v{latest_version}")
        
        # load model
        model = mlflow.sklearn.load_model(f"models:/{model_name}/{latest_version}")
        
        # Store in application state
        app.state.model = model
        app.state.model_version = latest_version
        app.state.model_run_id = run_id
        
        logger.info(f"Successfully loaded model {model_name} v{latest_version}")
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        app.state.model = None
        app.state.model_version = None
        app.state.model_run_id = None
        raise

@app.on_event("startup")
async def startup_event():
    """Initialize application state and load model"""
    app.state.client = MlflowClient()
    app.state.model_name = "Random_Forest_Regressor"
    app.state.model = None
    app.state.model_version = None
    app.state.model_run_id = None
    
    # Load the initial model
    try:
        await load_model()
    except Exception as e:
        logger.error(f"Failed to load model during startup: {str(e)}")
        # Continue startup even if model loading fails

@app.get("/")
async def serve_predict_dashboard():
    return FileResponse("static/index.html")

@app.get("/token-dashboard", include_in_schema=False)
async def token_dashboard():
    return FileResponse("static/token-dashboard.html")

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
        # Check if model is loaded
        if app.state.model is None:
            # Try to reload model
            await load_model()
            if app.state.model is None:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Model is not available"
                )
        
        # Prepare input data
        input_df = pd.DataFrame([request.dict()])
        
        # Make prediction using cached model
        prediction = app.state.model.predict(input_df)[0]
        price_segment = assign_price_segment(prediction)

        return {
            "predicted_price": prediction,
            "price_segment": price_segment,
            "model_name": app.state.model_name,
            "model_version": app.state.model_version,
            "status": "success"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/batch-predict", response_model=List[BatchPredictionResponse])
async def batch_predict(requests: List[PredictionRequest]):
    """Endpoint untuk prediksi batch"""
    try:
        # Check if model is loaded
        if app.state.model is None:
            # Try to reload model
            await load_model()
            if app.state.model is None:
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Model is not available"
                )
        
        # Prepare input data
        input_df = pd.DataFrame([r.dict() for r in requests])
        
        # Make predictions using cached model
        predictions = app.state.model.predict(input_df)
        
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
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
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
        training_datetime = datetime.fromtimestamp(run.info.start_time / 1000)
        
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
    
    # Check model status with detailed information
    model_status = "loaded" if app.state.model is not None else "not_loaded"
    
    # Determine overall health status
    overall_status = "healthy"
    if model_status == "not_loaded":
        overall_status = "degraded"
    elif mlflow_status == "disconnected":
        overall_status = "degraded"
    # Database disconnected is not critical for prediction, so we don't degrade for that
    
    return {
        "status": overall_status,
        "dependencies": {
            "mlflow": mlflow_status,
            "database": mlflow_db_status,
            "model": {
                "name": app.state.model_name,
                "version": app.state.model_version,
                "status": model_status
            }
        },
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "response_time_ms": (time.time() - start_time) * 1000
    }

@app.post("/model/update", response_model=ModelUpdateResponse, dependencies=[Depends(verify_token)])
async def update_model(request: ModelUpdateRequest):
    """Promote model Staging ke Production dan reload model"""
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
        
        # Reload model after successful promotion
        logger.info("Reloading model after update...")
        await load_model()
        
        result["message"] = f"Model {model_name} v{request.version} promoted from Staging to Production and reloaded"
        result["current_version"] = app.state.model_version
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
    """Rollback model Archived ke versi tertentu dan reload model"""
    try:
        model_name = app.state.model_name or request.model_name
        
        result = promote_model(
            client=app.state.client,
            model_name=model_name,
            version=request.version,
            source_stage="Archived",
            archive_previous=True
        )
        
        # Reload model after successful rollback
        logger.info("Reloading model after rollback...")
        await load_model()
        
        result["message"] = f"Model {model_name} v{request.version} rolled back from Archived to Production and reloaded"
        result["current_version"] = app.state.model_version
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

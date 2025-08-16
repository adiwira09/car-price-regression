from fastapi import FastAPI, HTTPException, status, Body, Depends, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.openapi.utils import get_openapi
from prometheus_fastapi_instrumentator import Instrumentator

from typing import List
import logging
import time

from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
from datetime import datetime

from dotenv import load_dotenv
load_dotenv()

from schemas import *
from utils import *
from auth import create_access_token, verify_token

# logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Custom OpenAPI schema function
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title="Car Price Prediction API",
        version="1.0.0",
        description="""
Features
- **Single Prediction**: Get price prediction for individual cars
- **Batch Prediction**: Process multiple car predictions efficiently
- **Model Management**: Update and rollback ML models seamlessly
- **Health Monitoring**: Real-time system health checks
- **Secure Authentication**: Token-based API access

Authentication
All prediction endpoints require Bearer token authentication. 
Use the `/token` endpoint to obtain an access token with your API key.
        """,
        routes=app.routes,
        tags=[
            {
                "name": "Authentication",
                "description": "Endpoints for obtaining and managing API access tokens"
            },
            {
                "name": "Predictions",
                "description": "Core prediction endpoints for single and batch car price predictions"
            },
            {
                "name": "Model Management",
                "description": "Endpoints for managing ML models, including updates and rollbacks"
            },
            {
                "name": "System Health",
                "description": "Health check and system monitoring endpoints"
            }
        ]
    )
    app.openapi_schema = openapi_schema
    return app.openapi_schema

# Initialize FastAPI with enhanced configuration
app = FastAPI(
    title="Car Price Prediction API",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    swagger_ui_parameters={
        "defaultModelsExpandDepth": 2,
        "defaultModelExpandDepth": 2,
        "displayOperationId": False,
        "displayRequestDuration": True,
        "docExpansion": "list",
        "filter": True,
        "showExtensions": True,
        "showCommonExtensions": True,
        "syntaxHighlight.theme": "tomorrow-night"
    }
)

# Set custom OpenAPI schema
app.openapi = custom_openapi

app.mount("/static", StaticFiles(directory="static"), name="static")

Instrumentator().instrument(app).expose(app)

async def load_model():
    """Load the latest production model into memory"""
    try:
        if not hasattr(app.state, "client") or app.state.client is None:
            logger.warning("MLflow client is not initialized")
            app.state.model = None
            app.state.model_version = None
            app.state.model_run_id = None
            return

        client = app.state.client
        model_name = app.state.model_name
        
        # Check if we already have this version loaded
        if hasattr(app.state, 'model_version') and app.state.model is not None:
            try:
                # Verify current version is still the latest
                latest_versions = client.get_latest_versions(model_name, stages=["Production"])
                if latest_versions and latest_versions[0].version == app.state.model_version:
                    logger.info(f"Model {model_name} v{app.state.model_version} already loaded")
                    return
            except Exception as e:
                logger.warning(f"Failed to check current model version: {str(e)}")
        
        # Load model using utility function
        model, version, run_id = load_model_from_mlflow(client, model_name)
        
        # Store in application state
        app.state.model = model
        app.state.model_version = version
        app.state.model_run_id = run_id
        
    except Exception as e:
        logger.error(f"Failed to load model: {str(e)}")
        app.state.model = None
        app.state.model_version = None
        app.state.model_run_id = None

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

@app.get("/",
         include_in_schema=False,
         tags=["Dashboard"])
async def serve_predict_dashboard():
    return FileResponse("static/index.html")

@app.get("/token-dashboard", 
         include_in_schema=False,
         tags=["Dashboard"])
async def token_dashboard():
    return FileResponse("static/token-dashboard.html")

@app.post("/token",
          tags=["Authentication"],
          summary="Get Access Token",
          description="""
# Obtain a Bearer token for API authentication.

**Required**: Valid API key in request body

**Returns**: JWT access token for subsequent API calls
          """,
          response_description="Access token and token type")
async def get_token(api_key: str = Body(..., embed=True)):
    """Endpoint untuk mendapatkan token akses"""
    if not validate_api_key(api_key):
        raise HTTPException(status_code=401, detail="Invalid API key")
    access_token = create_access_token(data={"sub": "api_client"})
    return {"access_token": access_token, "token_type": "bearer"}

@app.get("/validate-token",
         tags=["Authentication"],
         summary="Validate Access Token",
         description="""
# Validate if the provided Bearer token is still valid.

**Required**: Valid Bearer token in Authorization header

**Returns**: Simple validation status
         """,
         response_description="Token validation status")
async def validate_token(current_user: dict = Depends(verify_token)):
    """Endpoint untuk memvalidasi token akses"""
    return {"valid": True, "message": "Token is valid"}

@app.post("/predict", 
          response_model=PredictionResponse, 
          dependencies=[Depends(verify_token)],
          tags=["Predictions"],
          summary="Single Car Price Prediction",
          description="""
# Predict the price of a single car based on its features.

**Authentication Required**: Bearer token

**Input**: Car specifications

**Output**: Predicted price with confidence segment and model metadata
          """,
          response_description="Prediction result with price and metadata")
async def predict(request: PredictionRequest):
    """Endpoint untuk prediksi tunggal"""
    try:
        # Ensure model is loaded
        if not await ensure_model_loaded(app.state, load_model):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model is not available"
            )
        
        # Make prediction using utility function
        result = make_single_prediction(
            model=app.state.model,
            request_data=request.dict(),
            model_name=app.state.model_name,
            model_version=app.state.model_version
        )
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/batch-predict", 
          response_model=List[BatchPredictionResponse], 
          dependencies=[Depends(verify_token)],
          tags=["Predictions"],
          summary="Batch Car Price Predictions",
          description="""
# Predict prices for multiple cars in a single request.

**Authentication Required**: Bearer token

**Input**: Array of car specifications

**Output**: Array of predictions with individual IDs and metadata

**Performance**: Optimized for processing multiple predictions efficiently
          """,
          response_description="List of prediction results with IDs")
async def batch_predict(requests: List[PredictionRequest]):
    """Endpoint untuk prediksi batch"""
    try:
        # Ensure model is loaded
        if not await ensure_model_loaded(app.state, load_model):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model is not available"
            )
        
        # Make batch predictions using utility function
        requests_data = [r.dict() for r in requests]
        results = make_batch_predictions(
            model=app.state.model,
            requests_data=requests_data,
            model_name=app.state.model_name,
            model_version=app.state.model_version
        )
        
        return results
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction failed: {str(e)}"
        )

@app.post("/batch-predict-file",
          response_model=List[BatchPredictionResponse],
          dependencies=[Depends(verify_token)],
          tags=["Predictions"],
          summary="Batch Car Price Predictions via File",
          description="""
# Predict prices for multiple cars by uploading CSV or Excel file.

**Authentication Required**: Bearer token

**Input**: CSV or Excel file with car specifications

**Output**: Array of predictions with individual IDs and metadata
          """,
          response_description="List of prediction results with IDs")
async def batch_predict_file(file: UploadFile = File(...)):
    """Endpoint untuk prediksi batch melalui file CSV / Excel"""
    try:
        # Validate file extension
        if not validate_file_extension(file.filename):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File harus CSV atau Excel"
            )

        # Ensure model is loaded
        if not await ensure_model_loaded(app.state, load_model):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Model is not available"
            )

        # Read file to DataFrame using utility function
        df = read_file_to_dataframe(file.file, file.filename)

        # Make batch predictions
        requests_data = df.to_dict(orient='records')
        results = make_batch_predictions(
            model=app.state.model,
            requests_data=requests_data,
            model_name=app.state.model_name,
            model_version=app.state.model_version
        )
        
        return results

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch prediction via file failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction via file failed: {str(e)}"
        )

@app.get("/model/metadata", 
         response_model=ModelMetadataResponse,
         tags=["Model Management"],
         summary="Get Model Metadata",
         description="""
# Retrieve detailed metadata about the currently active model.

**Returns**: Model information including version, training date, metrics, and parameters
         """,
         response_description="Comprehensive model metadata")
async def get_model_metadata():
    """Mengambil metadata model aktif dari MLflow"""
    try:
        client = app.state.client
        model_name = app.state.model_name

        # Get model metadata using utility function
        metadata = get_model_metadata_from_mlflow(client, model_name)
        return metadata
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to fetch model metadata: {str(e)}"
        )
    
@app.get("/health", 
         response_model=HealthCheckResponse,
         tags=["System Health"],
         summary="System Health Check",
         description="""
# Health check for the API and all dependencies.

**Monitors**: 
- MLflow connection status
- Database connectivity
- Model loading status
- Overall system health

**Response Time**: Includes API response time measurement
         """,
         response_description="Detailed health status of all system components")
async def health_check():
    """Health check API dan dependencies"""
    start_time = time.time()
    
    mlflow_status = "connected" if check_mlflow_connection() else "disconnected"
    mlflow_db_status = "connected" if check_postgres_connection() else "disconnected"
    model_status = "loaded" if app.state.model is not None else "not_loaded"
    
    # Determine overall health status using utility function
    overall_status = determine_overall_health_status(model_status, mlflow_status)
    
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

@app.post("/model/load", 
          tags=["Model Management"],
          summary="Load Model by Name",
          description="""
# Load a model from MLflow by name without changing its stage.

**Use Case**: 
- Switch between different models (e.g., Random_Forest_Regressor ↔ Linear_Regression)
- Useful when models are in different registries but both have Production versions.

**Notes**:
- This does not promote or rollback models in MLflow.
- By default, loads the latest version in Production stage.
          """)
async def load_model_by_name(model_name: str = Body(..., embed=True), stage: str = Body("Production", embed=True)):
    """
    Load a specific model by name and stage without changing its stage in MLflow.
    """
    try:
        app.state.model_name = model_name

        # Load model using utility function
        model, version, run_id = load_model_from_mlflow(app.state.client, model_name, stage)
        
        # Store in application state
        app.state.model = model
        app.state.model_version = version
        app.state.model_run_id = run_id

        return {
            "message": f"Model {model_name} loaded successfully from {stage} stage",
            "model_name": app.state.model_name,
            "model_version": app.state.model_version
        }

    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        logger.error(f"Failed to load model by name: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to load model: {str(e)}"
        )

@app.post("/model/update", 
          response_model=ModelUpdateResponse, 
          dependencies=[Depends(verify_token)],
          tags=["Model Management"],
          summary="Update Production Model",
          description="""
# Promote a model from Staging to Production and reload the active model.

**Authentication Required**: Bearer token

**Process**:
1. Promotes specified model version from Staging to Production
2. Archives the previous Production model
3. Reloads the new model into memory

**Use Case**: Deploy new model versions to production
          """,
          response_description="Update confirmation with new model version")
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

@app.post("/model/rollback", 
          response_model=ModelUpdateResponse, 
          dependencies=[Depends(verify_token)],
          tags=["Model Management"],
          summary="Rollback to Previous Model",
          description="""
# Rollback to a previously archived model version and reload the active model.

**Authentication Required**: Bearer token

**Process**:
1. Promotes specified archived model version to Production
2. Archives the current Production model
3. Reloads the rollback model into memory

**Use Case**: Restore previous model version in case of issues
          """,
          response_description="Rollback confirmation with restored model version")
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

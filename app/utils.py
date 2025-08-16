from typing import Dict, List, Tuple, Any
from datetime import datetime

import psycopg2
from psycopg2 import OperationalError
import mlflow
import pandas as pd
import logging

import os
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

def promote_model(
        client,
        model_name: str,
        version: str,
        source_stage: str,
        archive_previous: bool = True
) -> Dict[str, any]:
    """
    Promote a model version to a specific stage.

    Args:
        model_name: Name of the MLflow registered model.
        version: Version to promote.
        source_stage: Current stage of the version ("Staging" or "Archived").
        archive_previous: Whether to archive the current Production version.
    
    Returns:
        Dictionary with promotion details for the API response.
    """
    # 1. Validate version exists in source_stage
    model_versions = client.search_model_versions(f"name='{model_name}'")
    source_versions = [v for v in model_versions if v.current_stage == source_stage]
    target_version = next((v for v in source_versions if v.version == version), None)
    
    if not target_version:
        raise ValueError(f"Version {version} not found in stage '{source_stage}'")

    # 2. Handle previous Production version
    prod_versions = client.get_latest_versions(model_name, stages=["Production"])
    previous_version = prod_versions[0].version if prod_versions else None

    if previous_version and archive_previous:
        client.transition_model_version_stage(
            name=model_name,
            version=previous_version,
            stage="Archived"
        )

    # 3. Promote the target version
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Production"
    )

    # 4. Get updated version info
    updated_version = client.get_model_version(model_name, version)

    return {
        "status": "success",
        "previous_version": previous_version,
        "new_version": version,
        "model_name": model_name,
        "mlflow_run_id": updated_version.run_id,
        "timestamp": datetime.utcnow().isoformat() + "Z",
        "details": {
            "source_stage": source_stage,
            "target_stage": "Production",
            "archived_previous": previous_version is not None and archive_previous
        }
    }

def check_mlflow_connection() -> bool:
    try:
        mlflow.search_experiments()
        return True
    except Exception:
        return False

def check_postgres_connection():
    try:
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_MLFLOW_HOST"),
            port=5432,
            dbname=os.getenv("POSTGRES_MLFLOW_DB"),
            user=os.getenv("POSTGRES_MLFLOW_USER"),
            password=os.getenv("POSTGRES_MLFLOW_PASSWORD")
        )
        conn.close()
        return True
    except OperationalError as e:
        print(f"Postgres connection error: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

def assign_price_segment(price, bins=[0, 10495, 19045, 45400]): # Low, Mid, High
    if price <= bins[1]:
        return 'Low'
    elif price <= bins[2]:
        return 'Mid'
    else:
        return 'High'

def load_model_from_mlflow(client, model_name: str, stage: str = "Production") -> Tuple[Any, str, str]:
    """
    Load model from MLflow registry.
    
    Args:
        client: MLflow client instance
        model_name: Name of the registered model
        stage: Stage to load from (default: Production)
    
    Returns:
        Tuple of (model, version, run_id)
    
    Raises:
        ValueError: If model not found in specified stage
        Exception: If model loading fails
    """
    latest_versions = client.get_latest_versions(model_name, stages=[stage])
    if not latest_versions:
        raise ValueError(f"Model {model_name} not found in {stage} stage")
        
    latest_version = latest_versions[0].version
    run_id = latest_versions[0].run_id
    
    logger.info(f"Loading model {model_name} v{latest_version} from {stage}")
    
    # Load model
    model = mlflow.sklearn.load_model(f"models:/{model_name}/{latest_version}")
    
    logger.info(f"Successfully loaded model {model_name} v{latest_version}")
    
    return model, latest_version, run_id

def validate_file_extension(filename: str, allowed_extensions: List[str] = ['.csv', '.xls', '.xlsx']) -> bool:
    """
    Validate if file has allowed extension.
    
    Args:
        filename: Name of the file
        allowed_extensions: List of allowed extensions
    
    Returns:
        True if valid, False otherwise
    """
    return any(filename.endswith(ext) for ext in allowed_extensions)

def read_file_to_dataframe(file, filename: str) -> pd.DataFrame:
    """
    Read uploaded file to pandas DataFrame.
    
    Args:
        file: File object from FastAPI
        filename: Name of the file
    
    Returns:
        pandas DataFrame
    
    Raises:
        ValueError: If file format is not supported
    """
    if filename.endswith('.csv'):
        return pd.read_csv(file)
    elif filename.endswith(('.xls', '.xlsx')):
        return pd.read_excel(file)
    else:
        raise ValueError("Unsupported file format")

def make_single_prediction(model, request_data: dict, model_name: str, model_version: str) -> Dict[str, Any]:
    """
    Make prediction for single request.
    
    Args:
        model: Loaded ML model
        request_data: Input data as dictionary
        model_name: Name of the model
        model_version: Version of the model
    
    Returns:
        Dictionary with prediction result
    """
    input_df = pd.DataFrame([request_data])
    prediction = model.predict(input_df)[0]
    price_segment = assign_price_segment(prediction)
    
    return {
        "predicted_price": prediction,
        "price_segment": price_segment,
        "model_name": model_name,
        "model_version": model_version,
        "status": "success"
    }

def make_batch_predictions(model, requests_data: List[dict], model_name: str, model_version: str) -> List[Dict[str, Any]]:
    """
    Make predictions for batch requests.
    
    Args:
        model: Loaded ML model
        requests_data: List of input data as dictionaries
        model_name: Name of the model
        model_version: Version of the model
    
    Returns:
        List of prediction results
    """
    input_df = pd.DataFrame(requests_data)
    predictions = model.predict(input_df)
    
    results = []
    for idx, pred in enumerate(predictions):
        price_segment = assign_price_segment(pred)
        results.append({
            "id": idx,
            "predicted_price": pred,
            "price_segment": price_segment,
            "model_name": model_name,
            "model_version": model_version,
            "status": "success"
        })
    
    return results

def get_model_metadata_from_mlflow(client, model_name: str) -> Dict[str, Any]:
    """
    Get model metadata from MLflow.
    
    Args:
        client: MLflow client instance
        model_name: Name of the registered model
    
    Returns:
        Dictionary with model metadata
    
    Raises:
        ValueError: If model not found
        Exception: If metadata retrieval fails
    """
    latest_versions = client.get_latest_versions(model_name, stages=["Production"])
    if not latest_versions:
        raise ValueError(f"Model {model_name} not found in Production stage")
    
    latest_version = latest_versions[0].version
    run = client.get_run(latest_versions[0].run_id)
    if not run:
        raise ValueError(f"Run for model version {latest_version} not found")
    
    training_datetime = datetime.fromtimestamp(run.info.start_time / 1000)
    
    return {
        "model_name": model_name,
        "version": latest_version,
        "training_date": training_datetime.strftime("%Y-%m-%d"),
        "metrics": run.data.metrics,
        "parameters": run.data.params
    }

def determine_overall_health_status(model_status: str, mlflow_status: str) -> str:
    """
    Determine overall health status based on component statuses.
    
    Args:
        model_status: Status of the model ("loaded" or "not_loaded")
        mlflow_status: Status of MLflow connection ("connected" or "disconnected")
    
    Returns:
        Overall health status ("healthy" or "degraded")
    """
    if model_status == "not_loaded" or mlflow_status == "disconnected":
        return "degraded"
    return "healthy"

async def ensure_model_loaded(app_state, load_model_func):
    """
    Ensure model is loaded, attempt to reload if not available.
    
    Args:
        app_state: FastAPI application state
        load_model_func: Function to load the model
    
    Returns:
        True if model is available, False otherwise
    
    Raises:
        Exception: If model loading fails
    """
    if app_state.model is None:
        await load_model_func()
        if app_state.model is None:
            return False
    return True

def validate_api_key(api_key: str) -> bool:
    """
    Validate API key against environment variable.
    
    Args:
        api_key: The API key to validate
    
    Returns:
        True if valid, False otherwise
    """
    return api_key == os.getenv("API_KEY")
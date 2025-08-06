from typing import Dict
from datetime import datetime

import psycopg2
from psycopg2 import OperationalError
import mlflow

import os
from dotenv import load_dotenv

load_dotenv()

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
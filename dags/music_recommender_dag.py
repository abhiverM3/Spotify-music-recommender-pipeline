from airflow import DAG
from airflow.providers.databricks.operators.databricks import DatabricksSubmitRunOperator
from datetime import datetime, timedelta

# ============================================================
# Music Recommender Pipeline DAG
# ============================================================

default_args = {
    "owner": "tata1maropolo",
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

with DAG(
    dag_id="music_recommender_pipeline",
    default_args=default_args,
    description="End to end music recommender pipeline",
    schedule="@daily",
    start_date=datetime(2026, 4, 4),
    catchup=False,
    tags=["music", "recommender", "medallion"],
) as dag:

    # Task 1 - Bronze ingestion
    bronze_ingestion = DatabricksSubmitRunOperator(
        task_id="bronze_ingestion",
        databricks_conn_id="databricks_default",
        json={
            "run_name": "bronze_ingestion",
            "tasks": [
                {
                    "task_key": "bronze_ingestion",
                    "notebook_task": {
                        "notebook_path": "/Users/abhiverelly@outlook.com/Spotify_project/Bronze_layer_ingestion",
                        "source": "WORKSPACE",
                    },
                    "environment_key": "default"
                }
            ],
            "environments": [
                {
                    "environment_key": "default",
                    "spec": {"client": "2"}
                }
            ]
        },
    )

    # Task 2 - Bronze data quality
    bronze_data_quality = DatabricksSubmitRunOperator(
        task_id="bronze_data_quality",
        databricks_conn_id="databricks_default",
        json={
            "run_name": "bronze_data_quality",
            "tasks": [
                {
                    "task_key": "bronze_data_quality",
                    "notebook_task": {
                        "notebook_path": "/Users/abhiverelly@outlook.com/Spotify_project/Data_quality",
                        "source": "WORKSPACE",
                    },
                    "environment_key": "default"
                }
            ],
            "environments": [
                {
                    "environment_key": "default",
                    "spec": {"client": "2"}
                }
            ]
        },
    )

    # Task 3 - Quarantine layer
    quarantine_layer = DatabricksSubmitRunOperator(
        task_id="quarantine_layer",
        databricks_conn_id="databricks_default",
        json={
            "run_name": "quarantine_layer",
            "tasks": [
                {
                    "task_key": "quarantine_layer",
                    "notebook_task": {
                        "notebook_path": "/Users/abhiverelly@outlook.com/Spotify_project/Quarantine_layer",
                        "source": "WORKSPACE",
                    },
                    "environment_key": "default"
                }
            ],
            "environments": [
                {
                    "environment_key": "default",
                    "spec": {"client": "2"}
                }
            ]
        },
    )

    # Task 4 - Silver transformation
    silver_transformation = DatabricksSubmitRunOperator(
        task_id="silver_transformation",
        databricks_conn_id="databricks_default",
        json={
            "run_name": "silver_transformation",
            "tasks": [
                {
                    "task_key": "silver_transformation",
                    "notebook_task": {
                        "notebook_path": "/Users/abhiverelly@outlook.com/Spotify_project/Silver_layer_ingestion",
                        "source": "WORKSPACE",
                    },
                    "environment_key": "default"
                }
            ],
            "environments": [
                {
                    "environment_key": "default",
                    "spec": {"client": "2"}
                }
            ]
        },
    )

    # Task 5 - Silver data quality
    silver_data_quality = DatabricksSubmitRunOperator(
        task_id="silver_data_quality",
        databricks_conn_id="databricks_default",
        json={
            "run_name": "silver_data_quality",
            "tasks": [
                {
                    "task_key": "silver_data_quality",
                    "notebook_task": {
                        "notebook_path": "/Users/abhiverelly@outlook.com/Spotify_project/Data_quality_silver",
                        "source": "WORKSPACE",
                    },
                    "environment_key": "default"
                }
            ],
            "environments": [
                {
                    "environment_key": "default",
                    "spec": {"client": "2"}
                }
            ]
        },
    )

    # Task 6 - Gold ML
    gold_ml = DatabricksSubmitRunOperator(
        task_id="gold_ml",
        databricks_conn_id="databricks_default",
        json={
            "run_name": "gold_ml",
            "tasks": [
                {
                    "task_key": "gold_ml",
                    "notebook_task": {
                        "notebook_path": "/Users/abhiverelly@outlook.com/Spotify_project/Gold_layer_ingestion",
                        "source": "WORKSPACE",
                    },
                    "environment_key": "default"
                }
            ],
            "environments": [
                {
                    "environment_key": "default",
                    "spec": {"client": "2"}
                }
            ]
        },
    )

    # Task 7 - Incremental merge
    incremental_merge = DatabricksSubmitRunOperator(
        task_id="incremental_merge",
        databricks_conn_id="databricks_default",
        json={
            "run_name": "incremental_merge",
            "tasks": [
                {
                    "task_key": "incremental_merge",
                    "notebook_task": {
                        "notebook_path": "/Users/abhiverelly@outlook.com/Spotify_project/Incremental_merge",
                        "source": "WORKSPACE",
                    },
                    "environment_key": "default"
                }
            ],
            "environments": [
                {
                    "environment_key": "default",
                    "spec": {"client": "2"}
                }
            ]
        },
    )

    # Task 8 - Skew detection
    skew_detection = DatabricksSubmitRunOperator(
        task_id="skew_detection",
        databricks_conn_id="databricks_default",
        json={
            "run_name": "skew_detection",
            "tasks": [
                {
                    "task_key": "skew_detection",
                    "notebook_task": {
                        "notebook_path": "/Users/abhiverelly@outlook.com/Spotify_project/Skew_detection",
                        "source": "WORKSPACE",
                    },
                    "environment_key": "default"
                }
            ],
            "environments": [
                {
                    "environment_key": "default",
                    "spec": {"client": "2"}
                }
            ]
        },
    )

    # Task 9 - Monitoring
    monitoring = DatabricksSubmitRunOperator(
        task_id="monitoring",
        databricks_conn_id="databricks_default",
        json={
            "run_name": "monitoring",
            "tasks": [
                {
                    "task_key": "monitoring",
                    "notebook_task": {
                        "notebook_path": "/Users/abhiverelly@outlook.com/Spotify_project/Monitoring",
                        "source": "WORKSPACE",
                    },
                    "environment_key": "default"
                }
            ],
            "environments": [
                {
                    "environment_key": "default",
                    "spec": {"client": "2"}
                }
            ]
        },
    )

    # Full pipeline order
    bronze_ingestion >> bronze_data_quality >> quarantine_layer >> silver_transformation >> silver_data_quality >> gold_ml >> incremental_merge >> skew_detection >> monitoring

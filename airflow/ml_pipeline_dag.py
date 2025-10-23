# airflow/dags/ml_pipeline_dag.py

from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import sys, os

# Ensure Airflow can find your src modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

# Import your pipeline components
from src.componets.data_injection import DataInjection, DataInjectionConfig
from src.componets.data_transformation import DataTransformationConfig, DataTransformation
from src.componets.data_validation import DataValidationConfig, DataValidation
from src.componets.model_trainer import ModelTrainer, ModelTrainerConfig
from src.utils.logger import logging


# -------------------------------
# Default DAG arguments
# -------------------------------
default_args = {
    'owner': 'airflow',
    'start_date': datetime(2025, 10, 23),
    'retries': 1,
}

# -------------------------------
# Define the DAG
# -------------------------------
with DAG(
    dag_id='ml_pipeline_dag',
    default_args=default_args,
    schedule_interval='@weekly',  # can be '@daily', '@weekly', etc.
    catchup=False,
    tags=['ml', 'training'],
    description="End-to-end ML pipeline automation DAG"
) as dag:

    # -------------------------------
    # Task functions
    # -------------------------------

    def ingest_data():
        logging.info("Starting data ingestion...")
        config = DataInjectionConfig()
        obj = DataInjection(config)
        obj.initiate_inject_and_save_data()
        logging.info("Data ingestion completed successfully.")

    def validate_data():
        logging.info("Starting data validation...")
        config = DataValidationConfig()
        obj = DataValidation(config)
        obj.initiate_data_validation()
        logging.info("Data validation completed successfully.")

    def preprocess_data():
        logging.info("Starting data transformation...")
        config = DataTransformationConfig()
        obj = DataTransformation(config)
        obj.initiate_data_transformation()
        logging.info("Data transformation completed successfully.")

    def train_evaluate_model():
        logging.info("Starting model training...")
        config = ModelTrainerConfig()
        obj = ModelTrainer(config)
        obj.initiate_models()
        logging.info("Model training completed successfully.")

    def push_to_dagshub():
        logging.info("Push Trained Artifacts Automatically to DagsHub")
        os.system("git add artifacts/models && git commit -m 'Auto-update model' && git push origin main")



    # -------------------------------
    # Define Airflow tasks
    # -------------------------------

    t1 = PythonOperator(task_id='data_ingestion', python_callable=ingest_data)
    t2 = PythonOperator(task_id='data_validation', python_callable=validate_data)
    t3 = PythonOperator(task_id='data_transformation', python_callable=preprocess_data)
    t4 = PythonOperator(task_id='model_training', python_callable=validate_data)
    t5 = PythonOperator(task_id='model_evaluation', python_callable=train_evaluate_model)

    # -------------------------------
    # Set task dependencies
    # -------------------------------
    t1 >>  t3 >> t2 >> t4 >> t5


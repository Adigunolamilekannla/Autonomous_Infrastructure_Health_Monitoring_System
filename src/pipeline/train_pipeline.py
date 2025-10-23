from src.componets.data_injection import DataInjection, DataInjectionConfig
from src.componets.data_transformation import DataTransformationConfig, DataTransformation
from src.componets.data_validation import DataValidationConfig, DataValidation
from src.componets.model_trainer import ModelTrainer, ModelTrainerConfig
from src.utils.logger import logging


class TrainPipeline:
    def __init__(self):
        self.data_ingestion_config = DataInjectionConfig()
        self.data_transformation_config = DataTransformationConfig()
        self.data_validation_config = DataValidationConfig()
        self.model_config = ModelTrainerConfig()

    def start_data_ingestion(self):
        data_ingestion = DataInjection(data_injection_config=self.data_ingestion_config)
        data_ingestion.initiate_inject_and_save_data()
      

    def start_data_transformation(self):
        data_transformation = DataTransformation(data_transform_config=self.data_transformation_config)
        data_transformation.initiate_transform_data()
 

    def start_data_validation(self):
        validate_data = DataValidation(config=self.data_validation_config)
        validate_data.initiate_data_validation()
     

    def start_model_training(self):
        model = ModelTrainer(self.model_config)
        model.initiate_models()
        print("✅ Model training completed successfully.")

    def run_pipeline(self):
        logging.info("🚀 Starting training pipeline...")
        self.start_data_ingestion()
        self.start_data_transformation()
        self.start_data_validation()
        self.start_model_training()
        logging.info("🎉 Training pipeline completed!")


if __name__ == "__main__":
    pipeline = TrainPipeline()
    pipeline.run_pipeline()

from src.componets.data_injection import DataInjection,DataInjectionConfig
from src.componets.data_transformation import DataTransformationConfig,DataTransformation
from src.componets.data_validation import DataValidationConfig,DataValidation
from src.componets.model_trainer import ModelTrainer,ModelTrainerConfig

data_ingestion_config = DataInjectionConfig()
data_ingestion = DataInjection(data_injection_config=data_ingestion_config)
data_ingestion.initiate_inject_and_save_data()


data_transformation_config = DataTransformationConfig()
data_transformation = DataTransformation(data_transform_config=data_transformation_config)
data_transformation.initiate_transform_data()

data_valid_config = DataValidationConfig()
validate_data = DataValidation(config=data_valid_config)
validate_data.initiate_data_validation()

model_config = ModelTrainerConfig()
model = ModelTrainer(model_config)
model.initiate_models()
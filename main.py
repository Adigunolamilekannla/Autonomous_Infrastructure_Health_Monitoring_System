from src.componets.data_injection import DataInjection,DataInjectionConfig


data_ingestion_config = DataInjectionConfig()
data_ingestion = DataInjection(data_injection_config=data_ingestion_config)
data_ingestion.initiate_inject_and_save_data()
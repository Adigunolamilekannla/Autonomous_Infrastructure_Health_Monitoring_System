import sys,os
import torch
import pandas as pd
from dataclasses import dataclass
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import  TensorDataset, random_split
from src.utils.exception import CustomException
from src.utils.logger import logging
from src.utils.dir_manager import MakeDirectory


# ==================================================
# Configuration Class
# ==================================================
@dataclass
class DataTransformationConfig:
    """This class stores all data transformation paths."""
    
    # Processed data paths
    processed_train_bridge_data: str = "artifacts/processed_datas/bridge_data/train_bridge_data.csv"
    processed_test_bridge_data: str = "artifacts/processed_datas/bridge_data/test_bridge_data.csv"

    processed_train_rul_data: str = "artifacts/processed_datas/rul_data/train_rul_data.csv"
    processed_test_rul_data: str = "artifacts/processed_datas/rul_data/test_rul_data.csv"

    processed_bridge_image_dataset: str = "artifacts/processed_datas/bridge_image_dataset/image_dataset.pt"

    # Transformed data paths
    transform_train_bridge_data: str = "artifacts/transformed_data/bridge_data/train_bridge_data.pt"
    transform_test_bridge_data: str = "artifacts/transformed_data/bridge_data/test_bridge_data.pt"
    transform_train_rul_data: str = "artifacts/transformed_data/rul_data/train_rul_data.csv"
    transform_test_rul_data: str = "artifacts/transformed_data/rul_data/test_rul_data.csv"
    transform_train_bridge_image_dataset: str = "artifacts/transformed_data/bridge_image_dataset/train_image_dataset.pt"
    transform_test_bridge_image_dataset: str = "artifacts/transformed_data/bridge_image_dataset/test_image_dataset.pt"

    validate_bridged_data:str = os.path.join("artifacts", "validate_bridged", "validate_bridged.csv")


# ==================================================
# Data Transformation Class
# ==================================================
class DataTransformation:
    def __init__(self, data_transform_config: DataTransformationConfig):
        try:
            logging.info("Initializing DataTransformation...")
            self.data_transform_config = data_transform_config
        except Exception as e:
            raise CustomException(e, sys.exc_info())

    # ==================================================
    # Main Data Transformation Function
    # ==================================================
    def transform_data(self):
        try:
            # ===========================
            # RUL DATA TRANSFORMATION
            # ===========================
            logging.info("Transforming RUL Data...")
            X_train_rul = pd.read_csv(self.data_transform_config.processed_train_rul_data)
            X_test_rul = pd.read_csv(self.data_transform_config.processed_test_rul_data)

            

            # Save transformed RUL tensors
            MakeDirectory(self.data_transform_config.transform_train_rul_data)
            MakeDirectory(self.data_transform_config.transform_test_rul_data)
            X_train_rul.to_csv(self.data_transform_config.transform_train_rul_data)
            X_test_rul.to_csv( self.data_transform_config.transform_test_rul_data)
            logging.info("Successfully transformed RUL Data.")

            # ===========================
            # BRIDGE DATA TRANSFORMATION
            # ===========================
            logging.info("Transforming Bridge Data...")
            X_train = pd.read_csv(self.data_transform_config.processed_train_bridge_data)
            X_test = pd.read_csv(self.data_transform_config.processed_test_bridge_data)

            # Label Encoding (fit on train only)
            le_material = LabelEncoder()
            le_design = LabelEncoder()
            X_train['Material'] = le_material.fit_transform(X_train['Material'])
            X_train['Bridge_Design'] = le_design.fit_transform(X_train['Bridge_Design'])
            X_test['Material'] = le_material.transform(X_test['Material'])
            X_test['Bridge_Design'] = le_design.transform(X_test['Bridge_Design'])

            # One-hot encoding for Weather_Conditions (fit on combined unique values)
            weather_dummies_train = pd.get_dummies(X_train['Weather_Conditions'], prefix='Weather', drop_first=True, dtype=int)
            weather_dummies_test = pd.get_dummies(X_test['Weather_Conditions'], prefix='Weather', drop_first=True, dtype=int)
            # Align columns
            weather_dummies_test = weather_dummies_test.reindex(columns=weather_dummies_train.columns, fill_value=0)
            X_train = pd.concat([X_train.drop('Weather_Conditions', axis=1), weather_dummies_train], axis=1)
            X_test = pd.concat([X_test.drop('Weather_Conditions', axis=1), weather_dummies_test], axis=1)

            # Date handling
            X_train['Maintenance_History'] = pd.to_datetime(X_train['Maintenance_History'])
            X_test['Maintenance_History'] = pd.to_datetime(X_test['Maintenance_History'])
            X_train['Maintenance_Year'] = X_train['Maintenance_History'].dt.year
            X_train['Maintenance_Month'] = X_train['Maintenance_History'].dt.month
            X_test['Maintenance_Year'] = X_test['Maintenance_History'].dt.year
            X_test['Maintenance_Month'] = X_test['Maintenance_History'].dt.month
            X_train.drop('Maintenance_History', axis=1, inplace=True)
            X_test.drop('Maintenance_History', axis=1, inplace=True)

            # Material composition percentages (robust lambda with error handling)
            def extract_percent(comp, material):
                try:
                    return int(comp.split(f'{material} ')[1].split('%')[0])
                except (IndexError, ValueError):
                    return 0

            X_train['Steel_%'] = X_train['Material_Composition'].apply(lambda x: extract_percent(x, 'Steel'))
            X_train['Concrete_%'] = X_train['Material_Composition'].apply(lambda x: extract_percent(x, 'Concrete'))
            X_train['Wood_%'] = X_train['Material_Composition'].apply(lambda x: extract_percent(x, 'Wood'))
            X_test['Steel_%'] = X_test['Material_Composition'].apply(lambda x: extract_percent(x, 'Steel'))
            X_test['Concrete_%'] = X_test['Material_Composition'].apply(lambda x: extract_percent(x, 'Concrete'))
            X_test['Wood_%'] = X_test['Material_Composition'].apply(lambda x: extract_percent(x, 'Wood'))

            X_train.drop(['Material_Composition', 'Construction_Quality'], axis=1, inplace=True)
            X_test.drop(['Material_Composition', 'Construction_Quality'], axis=1, inplace=True)

            # Target mapping
            X_train['Collapse_Status'] = X_train['Collapse_Status'].map({'Standing': 0, 'Collapsed': 1})
            X_test['Collapse_Status'] = X_test['Collapse_Status'].map({'Standing': 0, 'Collapsed': 1})
            MakeDirectory(self.data_transform_config.validate_bridged_data)
            X_train.to_csv(self.data_transform_config.validate_bridged_data,index=False)

            y_train = X_train['Collapse_Status']
            y_test = X_test['Collapse_Status']
            X_train.drop('Collapse_Status', axis=1, inplace=True)
            X_test.drop('Collapse_Status', axis=1, inplace=True)

            # Standardization
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)

            X_train_tensor = torch.tensor(X_train_scaled).float()
            X_test_tensor = torch.tensor(X_test_scaled).float()
            y_train_tensor = torch.tensor(y_train.values).float().unsqueeze(1)
            y_test_tensor = torch.tensor(y_test.values).float().unsqueeze(1)

            # Tensor Datasets
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

            # Save transformed bridge datasets (not loaders)
            MakeDirectory(self.data_transform_config.transform_train_bridge_data)
            MakeDirectory(self.data_transform_config.transform_test_bridge_data)
            torch.save(train_dataset, self.data_transform_config.transform_train_bridge_data)
            torch.save(test_dataset, self.data_transform_config.transform_test_bridge_data)
            logging.info("Successfully transformed Bridge Data.")

            # ===========================
            # BRIDGE IMAGE DATA TRANSFORMATION
            # ===========================
            logging.info("Transforming Bridge Image Data...")
            full_dataset = torch.load(self.data_transform_config.processed_bridge_image_dataset)
            train_size = int(0.8 * len(full_dataset))
            test_size = len(full_dataset) - train_size
            train_dataset_img, test_dataset_img = random_split(full_dataset, [train_size, test_size])

            # Create directories
            MakeDirectory(self.data_transform_config.transform_train_bridge_image_dataset)
            MakeDirectory(self.data_transform_config.transform_test_bridge_image_dataset)

            # Save datasets (not loaders)
            torch.save(train_dataset_img, self.data_transform_config.transform_train_bridge_image_dataset)
            torch.save(test_dataset_img, self.data_transform_config.transform_test_bridge_image_dataset)
            logging.info("Successfully transformed Bridge Image Data.")

        except Exception as e:
            raise CustomException(e, sys.exc_info())

    # ==================================================
    # Entry Function
    # ==================================================
    def initiate_transform_data(self):
        try:
            logging.info("Initializing Data Transformation...")
            self.transform_data()
            logging.info("Data Transformation Completed Successfully.")
        except Exception as e:
            raise CustomException(e, sys.exc_info())
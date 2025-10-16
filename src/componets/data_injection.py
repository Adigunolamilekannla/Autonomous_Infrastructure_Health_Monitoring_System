import sys
import gc
import torch
import pandas as pd
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from src.utils.exception import CustomException
from src.utils.logger import logging
from src.utils.dir_manager import MakeDirectory


@dataclass
class DataInjectionConfig:
    """
    Stores all file paths for data ingestion and saving.
    """
    raw_bridge_data: str = "data/brigde_dataset/bridge_data.csv"
    raw_rul_data: str = "data/RUL_dataset/rul_dataset_realistic.csv"
    raw_bridge_image_dataset: str = "data/image_dataset/road_image_dataset"

    processed_train_bridge_data: str = "artifacts/processed_datas/bridge_data/train_bridge_data.csv"
    processed_test_bridge_data: str = "artifacts/processed_datas/bridge_data/test_bridge_data.csv"

    processed_train_rul_data: str = "artifacts/processed_datas/rul_data/train_rul_data.csv"
    processed_test_rul_data: str = "artifacts/processed_datas/rul_data/test_rul_data.csv"

    processed_bridge_image_dataset: str = "artifacts/processed_datas/bridge_image_dataset/image_dataset.pt"


class SafeImageFolder(datasets.ImageFolder):
    def __getitem__(self, index):
        try:
            return super().__getitem__(index)
        except Exception as e:
            logging.warning(f"Skipping corrupted image at index {index}: {e}")
            return None



class DataInjection:
    def __init__(self, data_injection_config: DataInjectionConfig):
        try:
            logging.info("Initializing DataInjectionConfig")
            self.data_injection_config = data_injection_config
        except Exception as e:
            raise CustomException(e, sys)

    def inject_and_save_data(self):
        try:
            # ================================
            # 1️⃣ RUL DATA INGESTION
            # ================================
            logging.info("Ingesting RUL Data...")
            rul_df = pd.read_csv(self.data_injection_config.raw_rul_data)

            # Drop unnecessary columns
            rul_df = rul_df.drop(columns=["Machine_ID", "Failure_Type"], errors="ignore")

            X, y = rul_df.drop(columns=["RUL"]), rul_df["RUL"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            # Add target column back for convenience
            X_train["RUL"] = y_train
            X_test["RUL"] = y_test

            # Ensure directories exist
            MakeDirectory(self.data_injection_config.processed_train_rul_data)
            MakeDirectory(self.data_injection_config.processed_test_rul_data)

            # Save data
            X_train.to_csv(self.data_injection_config.processed_train_rul_data, index=False)
            X_test.to_csv(self.data_injection_config.processed_test_rul_data, index=False)
            logging.info("✅ Successfully saved RUL data.")

            # ================================
            # 2️⃣ BRIDGE SENSOR DATA INGESTION
            # ================================
            logging.info("Ingesting Bridge Sensor Data...")
            df = pd.read_csv(self.data_injection_config.raw_bridge_data)

            # Drop irrelevant columns safely
            df = df.drop(columns=["Location", "Bridge_ID"], errors="ignore")

            X = df.drop(columns=["Collapse_Status"])
            y = df["Collapse_Status"]
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

            X_train["Collapse_Status"] = y_train
            X_test["Collapse_Status"] = y_test

            MakeDirectory(self.data_injection_config.processed_train_bridge_data)
            MakeDirectory(self.data_injection_config.processed_test_bridge_data)

            X_train.to_csv(self.data_injection_config.processed_train_bridge_data, index=False)
            X_test.to_csv(self.data_injection_config.processed_test_bridge_data, index=False)
            logging.info("✅ Successfully saved Bridge Sensor Data.")

            # ================================
            # 3️⃣ BRIDGE IMAGE DATA INGESTION
            # ================================
            logging.info("Ingesting Bridge Image Data...")

            


            transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

            dataset = SafeImageFolder(
            root=self.data_injection_config.raw_bridge_image_dataset,
            transform=transform
            )

            dataset = SafeImageFolder(root=self.data_injection_config.raw_bridge_image_dataset,
                                      transform=transform)

            MakeDirectory(self.data_injection_config.processed_bridge_image_dataset)

            torch.save(dataset, self.data_injection_config.processed_bridge_image_dataset)
            logging.info("✅ Successfully saved Bridge Image Data.")

            gc.collect()
            torch.cuda.empty_cache()

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_inject_and_save_data(self):
        try:
            logging.info("🚀 Starting Data Injection Process")
            self.inject_and_save_data()
        except Exception as e:
            raise CustomException(e, sys)





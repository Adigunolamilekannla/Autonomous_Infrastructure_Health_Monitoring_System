import os
import sys
import pandas as pd
from dataclasses import dataclass
from src.utils.exception import CustomException
from src.utils.logger import logging
from src.utils.dir_manager import MakeDirectory
from src.utils.load_save import load_yaml


@dataclass
class DataValidationConfig:
    schema_path: str = os.path.join("config", "data_schema.yaml")  
    report_path: str = os.path.join("artifacts", "data_valid_report", "data_validation_report.txt")
    rul_data_path: str = os.path.join("artifacts", "transformed_data", "rul_data", "train_rul_data.csv")
    bridge_sensor_data_path: str = os.path.join("artifacts", "validate_bridged", "validate_bridged.csv")


class DataValidation:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def load_data(self, data_path: str) -> pd.DataFrame:
        try:
            if not os.path.exists(data_path):
                raise FileNotFoundError(f"Dataset not found at: {data_path}")
            logging.info(f"✅ Loading dataset from {data_path}")
            return pd.read_csv(data_path)
        except Exception as e:
            raise CustomException(e, sys)

    # ---------------- RUL DATA VALIDATION ----------------

    def validate_rul_data_columns(self, df: pd.DataFrame, schema: dict):
        """Check if RUL data has correct columns"""
        target_name = schema["rul_data"]["target"]["name"]
        if target_name in df.columns:
            df = df.drop(target_name, axis=1)

        schema_cols = [col["name"] for col in schema["rul_data"]["columns"]]
        missing = [c for c in schema_cols if c not in df.columns]
        extra = [c for c in df.columns if c not in schema_cols]
        return missing, extra

    def validate_rul_data_dtypes(self, df: pd.DataFrame, schema: dict):
        """Check RUL data types"""
        target_name = schema["rul_data"]["target"]["name"]
        if target_name in df.columns:
            df = df.drop(target_name, axis=1)

        mismatched = {}
        for col_def in schema["rul_data"]["columns"]:
            name, expected_type = col_def["name"], col_def["dtype"]
            if name in df.columns:
                actual_type = str(df[name].dtype)
                if expected_type not in actual_type:
                    mismatched[name] = {"expected": expected_type, "found": actual_type}
        return mismatched

    # ---------------- BRIDGE SENSOR DATA VALIDATION ----------------

    def validate_bridge_sensor_data_columns(self, df: pd.DataFrame, schema: dict):
        """Check if bridge sensor data has correct columns"""
        target_name = schema["bridge_data"]["target"]["name"]
        if target_name in df.columns:
            df = df.drop(target_name, axis=1)

        schema_cols = [col["name"] for col in schema["bridge_data"]["columns"]]
        missing = [c for c in schema_cols if c not in df.columns]
        extra = [c for c in df.columns if c not in schema_cols]
        return missing, extra

    def validate_bridge_sensor_data_dtypes(self, df: pd.DataFrame, schema: dict):
        """Check bridge sensor data types"""
        target_name = schema["bridge_data"]["target"]["name"]
        if target_name in df.columns:
            df = df.drop(target_name, axis=1)

        mismatched = {}
        for col_def in schema["bridge_data"]["columns"]:
            name, expected_type = col_def["name"], col_def["dtype"]
            if name in df.columns:
                actual_type = str(df[name].dtype)
                if expected_type not in actual_type:
                    mismatched[name] = {"expected": expected_type, "found": actual_type}
        return mismatched

    # ---------------- COMMON VALIDATION ----------------

    def validate_missing_values(self, df: pd.DataFrame):
        """Check for missing or NaN values"""
        missing_counts = df.isnull().sum()
        return missing_counts[missing_counts > 0].to_dict()

    def generate_report(self, results: dict):
        """Save validation report"""
        try:
            MakeDirectory(self.config.report_path)
            with open(self.config.report_path, "w") as f:
                for key, value in results.items():
                    f.write(f"{key}: {value}\n\n")
            logging.info(f"📄 Data validation report saved at: {self.config.report_path}")
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_validation(self):
        """Run the full data validation pipeline"""
        logging.info("🚀 Data validation started...")
        try:
            # Load datasets
            rul_df = self.load_data(self.config.rul_data_path)
            bridge_df = self.load_data(self.config.bridge_sensor_data_path)

            # Load schema
            schema = load_yaml(self.config.schema_path)

            # --- RUL data checks ---
            missing_rul, extra_rul = self.validate_rul_data_columns(rul_df, schema)
            mismatched_rul = self.validate_rul_data_dtypes(rul_df, schema)
            missing_values_rul = self.validate_missing_values(rul_df)

            # --- Bridge sensor data checks ---
            missing_bridge, extra_bridge = self.validate_bridge_sensor_data_columns(bridge_df, schema)
            mismatched_bridge = self.validate_bridge_sensor_data_dtypes(bridge_df, schema)
            missing_values_bridge = self.validate_missing_values(bridge_df)

            # --- Combine results ---
            results = {
                "missing_columns_rul_data": missing_rul,
                "extra_columns_rul_data": extra_rul,
                "dtype_mismatches_rul_data": mismatched_rul,
                "missing_values_rul_data": missing_values_rul,
                "duplicate_rows_rul_data": rul_df.duplicated().sum(),

                "missing_columns_bridge_data": missing_bridge,
                "extra_columns_bridge_data": extra_bridge,
                "dtype_mismatches_bridge_data": mismatched_bridge,
                "missing_values_bridge_data": missing_values_bridge,
                "duplicate_rows_bridge_data": bridge_df.duplicated().sum(),
            }

            # Save report
            self.generate_report(results)
            logging.info("✅ Data validation completed successfully.")
            return results

        except Exception as e:
            raise CustomException(e, sys)

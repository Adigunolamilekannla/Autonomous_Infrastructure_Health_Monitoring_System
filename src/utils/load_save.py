import yaml
import sys
import os
from src.utils.exception import CustomException
from src.utils.logger import logging
from src.utils.dir_manager import MakeDirectory


def load_yaml(file_path):
    """Load schema from YAML"""
    logging.info(f"Loading yaml file from {file_path}")
    try:
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                return yaml.safe_load(f)
        else:
            raise CustomException("File Path Doesn't Exist", sys)
    except Exception as e:
        raise CustomException(e, sys)


# ------------------------------------------------------
# ✅ Save Model Evaluation Report
# ------------------------------------------------------
def generate_report(results: dict):
    try:
        # Define report path
        model_evaluation_report = os.path.join("artifacts", "reports", "model_evaluation_report.txt")

        # Ensure parent directory exists
        
        MakeDirectory(model_evaluation_report)

        # Write report
        with open(model_evaluation_report, "a") as f:
            for section, metrics in results.items():
                f.write(f"\n{section.upper()} METRICS\n")
                f.write("-" * 50 + "\n")
                for key, value in metrics.items():
                    f.write(f"{key}: {value}\n")
                f.write("\n")

        logging.info(f"📄 Model Evaluation report saved at: {model_evaluation_report}")

    except Exception as e:
        raise CustomException(e, sys)

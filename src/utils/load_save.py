import yaml
from src.utils.exception import CustomException
from src.utils.logger import logging
import sys,os




def load_yaml(file_path):
        """Load schema from YAML"""
        logging.info(f"LOading yaml file from {file_path}")
        try:
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    return yaml.safe_load(f)
            else:
                 raise CustomException("File Path Didn't Exists", sys)
        except Exception as e:
            raise CustomException(e, sys)
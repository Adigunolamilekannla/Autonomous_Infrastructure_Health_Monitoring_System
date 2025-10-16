import os
import sys
from src.utils.logger import logging
from src.utils.exception import CustomException


class MakeDirectory:
    def __init__(self, dir_path: str):
        try:
            logging.info(f"Creating directory: {dir_path}")
    
            os.makedirs(os.path.dirname(dir_path), exist_ok=True)
    
        except Exception as e:
            raise CustomException(e, sys)

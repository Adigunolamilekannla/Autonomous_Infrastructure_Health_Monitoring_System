import sys
from src.utils.logger import logging


class CustomException(Exception):
    """
    Custom Exception class for the Network Security project.
    It captures the error message, line number, and filename
    where the exception occurred.
    """

    def __init__(self, error_message: str, error_details: sys):
        """
        Constructor to initialize the custom exception.

        Parameters
        ----------
        error_message : str
            The error message from the exception.

        error_details : sys
            The sys module is passed to extract detailed
            information about the exception (line number, filename).
        """
        super().__init__(error_message)  # Initialize base Exception
        self.error_message = str(error_message)

        # Extract traceback info
        _, _, exc_tb = error_details.exc_info()
        self.lineno = exc_tb.tb_lineno if exc_tb else None
        self.filename = exc_tb.tb_frame.f_code.co_filename if exc_tb else None

        # Log immediately when exception is created
        logging.error(
            f"NetworkSecurityException initialized: {self.error_message} "
            f"(File: {self.filename}, Line: {self.lineno})"
        )

    def __str__(self):
        """
        Return a string representation of the exception
        including filename, line number, and error message.
        """
        error_msg = (
            f"Error occurred in Python script [{self.filename}] "
            f"at line number [{self.lineno}] "
            f"with error message: [{self.error_message}]"
        )
        logging.error(error_msg)  # Log when converting to string
        return error_msg

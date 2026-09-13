import logging
import numpy as nm
from sklearn.preprocessing import StandardScaler
from src.exception import DataValidationError

logging.basicConfig(level=logging.INFO , format="%(asctime)s - [%(levelname)s] - %(message)s")
logger = logging.getLogger(__name__)


class creditprocessor:
    def __init__(self):
        self.scaler = StandardScaler()
        logger.info("CreditDataPreorocessor intialized cleanely")

    def fit_pipeline(self, x_train):
        self._validate_input(x_train)
        self.scaler.fit_transform(x_train)
        logger.info("preprocessing scaling passed")
        return self

    def transform_data(self, x):
        self._validate_input(x)
        scaled_data = self.scaler.transform(x)
        logger.info("Successfully transform input feature vector")
        return scaled_data

    def _validate_input(self, x):
        if not isinstance(x, nm.ndarray):
            logger.error("Data validation failed: Input")
            raise DataValidationError("Not an array")

        if x.size == 0:
            logger.error("Empty array")
            raise DataValidationError("Array is Empty")

        if x.ndim != 2:
            logger.error(f"Data validation{x.ndim}D array.")
            raise DataValidationError(f"Expected 2d matrix,got {x.ndim}D.")


if __name__ == "__main__":
    x_historic = nm.array([[750, 0.2], [500, 0.6], [620, 0.4]])
    pipeline_processor = creditprocessor()
    pipeline_processor.fit_pipeline(x_historic)
    x_scaled = pipeline_processor.transform_data(x_historic)
    print(x_historic)
    print(x_scaled)

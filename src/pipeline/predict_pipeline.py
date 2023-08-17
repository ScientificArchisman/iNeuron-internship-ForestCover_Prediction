import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.logger import logging
from src.utils import load_pickle
from src.components import data_transformation



class Predictpipeline:
    def __init__(self) -> None:
        pass

    def predict(self, data):
        """Predicts the data
        Args:
            data (pd.DataFrame): data to be predicted
        Returns:
            prediction: prediction of the data  
        """
        # Read the data
        # get model
        try:
            model = load_pickle("artifacts/model_data/model.pkl")
        except Exception as e:
            logging.error(f"Error in loading model: {e}")
            raise e

        # get preprocessor and fit transform data
        try:
            preprocessor = data_transformation.DataTransformation()
            feature_preprocessor, _ = preprocessor.get_data_preprocessor()
        except Exception as e:
            logging.error(f"Error in getting preprocessor: {e}")
            raise e
        
        # transform data
        try:
            feature_preprocessor.fit(data) # fit on test data
            data = feature_preprocessor.transform(data)
        except Exception as e:
            logging.error(f"Error in preprocessing data: {e}")
            raise e

        # Make predictions and print
        try:
            prediction = model.predict(data)
            return prediction

        except Exception as e:
            logging.error(f"Error in making predictions: {e}")
            raise e
    



if __name__ == "__main__":
    data = pd.read_csv('artifacts/data/test_data.csv')
    logging.info("Data loaded successfully")
    pipeline = Predictpipeline()
    prediction = pipeline.predict(data)
    logging.info("Prediction completed")
    print(prediction)
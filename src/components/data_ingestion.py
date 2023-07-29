import os
import pandas as pd
import sys
from dataclasses import dataclass
from sklearn.model_selection import train_test_split
from src.exception import CustomException
from src.logger import logging


    
@dataclass
class DataIngestion:
    root_data_file : str = os.path.join("artifacts")
    test_data_file : str = os.path.join(root_data_file, "test_data.csv")
    train_data_file : str = os.path.join(root_data_file, "train_data.csv")
    raw_data_file : str = os.path.join(root_data_file, "raw_data.csv")
    
    logging.info("Entering data ingestion")
    def initiate_data_split(self):
        os.makedirs(self.root_data_file, exist_ok=True)
        
        try:
            raw_data = pd.read_csv("train_forest.csv")
        except Exception as e:  
            logging.error("Error reading data")
            raise CustomException(e, sys)
        
        logging.info("Splitting data into train and test")
        train_data, test_data = train_test_split(raw_data, test_size=0.2, random_state=42)

        try:
            logging.info("Saving data to artifacts")
            train_data.to_csv(self.train_data_file, index=False)
            test_data.to_csv(self.test_data_file, index=False)
            raw_data.to_csv(self.raw_data_file, index=False)
        except Exception as e:
            logging.error("Error saving data to artifacts")
            raise CustomException(e, sys)


        return self.train_data_file, self.test_data_file, self.raw_data_file
    

if __name__ == "__main__":
    data_ingestion = DataIngestion()
    data_ingestion.initiate_data_split()

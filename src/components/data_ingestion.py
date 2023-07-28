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
    
    def initiate_data_split(self):
        os.makedirs(self.root_data_file, exist_ok=True)

        try:
            raw_data = pd.read_csv("train_forest.csv")
        except Exception as e:  
            raise CustomException(e, sys)
        
        train_data, test_data = train_test_split(raw_data, test_size=0.2, random_state=42)

        try:
            train_data.to_csv(self.train_data_file, index=False)
            test_data.to_csv(self.test_data_file, index=False)
            raw_data.to_csv(self.raw_data_file, index=False)
        except Exception as e:
            raise CustomException(e, sys)

        return self.train_data_file, self.test_data_file, self.raw_data_file
    

if __name__ == "__main__":
    data_ingestion = DataIngestion()
    data_ingestion.initiate_data_split()

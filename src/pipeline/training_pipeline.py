import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from src.logger import logging
from src.components import data_transformation
from src.components.model_training import ModelTraining
from src.components.data_ingestion import DataIngestion

class TrainModel:
    def __init__(self, data_path) -> None:
         """ 
         This function initializes the class
         :param data_path: Path to the data
         :return: None"""
         self.data_path = data_path
         self.data = pd.read_csv(data_path)

    def ingest_data(self):
        data_ingestion = DataIngestion(self.data_path)
        train_file, test_file, raw_file = data_ingestion.initiate_data_split()
        return train_file, test_file, raw_file

    
    def transform_data(self):
        train_file, test_file, raw_file = self.ingest_data()
        data_transformation_obj = data_transformation.DataTransformation()
        X_train, X_test, y_train, y_test = data_transformation_obj.initiate_data_transformation(train_file, test_file)
        return X_train, X_test, y_train, y_test
    
    
    def train_model(self):
        self.transform_data()
        model_training = ModelTraining(X_train_path= "artifacts/transformed_data/train_features.csv", 
                              X_test_path= "artifacts/transformed_data/test_features.csv", 
                              y_train_path= "artifacts/transformed_data/train_labels.csv",
                              y_test_path= "artifacts/transformed_data/test_labels.csv")
        _, model, _ = model_training.fit_models()
        return model
    
    

if __name__ == "__main__":
    train_model = TrainModel(data_path="artifacts/data/raw_data.csv")
    model = train_model.train_model()
    print(model)
    
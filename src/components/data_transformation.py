from src.logger import logging
from src.exception import CustomException
from src.utils import save_pickle
import numpy as np
import sys
import pandas as pd
import os
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer

class DataTransformation:

    def __init__(self):
        self.PREPROCESSOR_PATH : str = os.path.join("artifacts", "preprocessor.pkl")
        self.numerical_columns : list = ['Elevation', 'Aspect', 'Slope',
                                'Horizontal_Distance_To_Hydrology', 'Vertical_Distance_To_Hydrology',
                                'Horizontal_Distance_To_Roadways', 'Hillshade_9am', 'Hillshade_Noon',
                                'Hillshade_3pm', 'Horizontal_Distance_To_Fire_Points']
        self.categorical_columns : list = ['Wilderness_Area1', 'Wilderness_Area2', 'Wilderness_Area3',
                                'Wilderness_Area4', 'Soil_Type1', 'Soil_Type2', 'Soil_Type3',
                                'Soil_Type4', 'Soil_Type5', 'Soil_Type6', 'Soil_Type7', 'Soil_Type8',
                                'Soil_Type9', 'Soil_Type10', 'Soil_Type11', 'Soil_Type12',
                                'Soil_Type13', 'Soil_Type14', 'Soil_Type15', 'Soil_Type16',
                                'Soil_Type17', 'Soil_Type18', 'Soil_Type19', 'Soil_Type20',
                                'Soil_Type21', 'Soil_Type22', 'Soil_Type23', 'Soil_Type24',
                                'Soil_Type25', 'Soil_Type26', 'Soil_Type27', 'Soil_Type28',
                                'Soil_Type29', 'Soil_Type30', 'Soil_Type31', 'Soil_Type32',
                                'Soil_Type33', 'Soil_Type34', 'Soil_Type35', 'Soil_Type36',
                                'Soil_Type37', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40']
        
        self.root_data_file : str = os.path.join("artifacts", "transformed_data")
        self.train_features_path : str =  os.path.join(self.root_data_file, "train_features.csv")
        self.train_labels_path : str = os.path.join(self.root_data_file, "train_labels.csv")
        self.test_features_path : str = os.path.join(self.root_data_file, "test_features.csv")
        self.test_labels_path : str = os.path.join(self.root_data_file, "test_labels.csv")
    
    def get_data_preprocessor(self, k:int = 15):
        """Returns the preprocessor object and saves it in pkl format
        Args:
            k (int, optional): Number of features to select according to Mutual information. Defaults to 15.
        Returns:
            preprocessor: sklearn ColumnTransformer object
        """
        # Numerical pipeline
        numerical_pipeline = Pipeline(
            steps = [("imputer", SimpleImputer(strategy = "mean")),
                    ("scaler", StandardScaler())])
        
        # Categorical pipeline
        categorical_pipeline = Pipeline(
            steps = [("imputer", SimpleImputer(strategy = "most_frequent")),
                     ("onehot", OneHotEncoder(handle_unknown = "ignore"))])
        
        # Column transformer
        transformer = ColumnTransformer([("num_pipeline", numerical_pipeline, self.numerical_columns), 
                                         ("cat_pipeline", categorical_pipeline, self.categorical_columns)])
        
        # Mutual information feature selection
        overall_pipeline = Pipeline(steps = [("transformer", transformer), 
                                             ("remover", SelectKBest(score_func=mutual_info_classif, k = k))])  

        # Save the preprocessor
        save_pickle(overall_pipeline, self.PREPROCESSOR_PATH) 
        return overall_pipeline 

    def initiate_data_transformation(self, train_data_file : str, test_data_file : str):
        """Transforms the data and saves it in artifacts
        Args:
            train_data_file (str): Path to train data
            test_data_file (str): Path to test data
        Returns:
            X_train (np.ndarray): Transformed training features
            X_test (np.ndarray): Transformed testing features
            y_train (np.ndarray): Transformed training labels
            y_test (np.ndarray): Transformed testing labels
        """
        try:
            train_data = pd.read_csv(train_data_file).drop(columns = ["Id"])
            test_data = pd.read_csv(test_data_file).drop(columns = ["Id"])
            logging.info("Data loaded successfully for transformation")
        except Exception as e:
            logging.error(f"Error loading data for transformation - {e}")
            raise CustomException(e, sys)

        # Split data into features and labels
        X_train = train_data.drop(columns = ["Cover_Type"])
        y_train = train_data["Cover_Type"]

        X_test = test_data.drop(columns = ["Cover_Type"])
        y_test = test_data["Cover_Type"]

        preprocessor = self.get_data_preprocessor() # Get the preprocessor

        # Fit the training set with the preprocessor
        preprocessor.fit(X_train, y_train)

        # Transform the training and testing dataset with the fitted preprocessor
        X_train = preprocessor.transform(X_train)
        X_test = preprocessor.transform(X_test)

        # Save the transformed data
        os.makedirs(self.root_data_file, exist_ok=True)
        pd.DataFrame(X_train).to_csv(self.train_features_path, index = False)
        pd.DataFrame(y_train).to_csv(self.train_labels_path, index = False)
        pd.DataFrame(X_test).to_csv(self.test_features_path, index = False)
        pd.DataFrame(y_test).to_csv(self.test_labels_path, index = False)

        return (X_train, X_test, y_train, y_test)




    def __str__(self):
        return f"DataTransformation class object created at {self.PREPROCESSOR_PATH}"
    


if __name__ == "__main__":
    data_transform = DataTransformation()
    data_transform.get_data_preprocessor()
    data_transform.initiate_data_transformation("artifacts/train_data.csv", "artifacts/test_data.csv")

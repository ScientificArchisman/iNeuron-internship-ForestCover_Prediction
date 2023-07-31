import numpy as np
import pandas as pd
import os
import sys
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from src.exception import CustomException
from src.logger import logging
from src.utils import evaluate_models, save_pickle

class ModelTraining:
    def __init__(self):
        self.root_model_file = os.path.join("artifacts", "model_data")
        self.model_results_file = os.path.join(self.root_model_file, "model_results.csv")
        self.model_file = os.path.join(self.root_model_file, "model.pkl")

    def fit_models(self, X_train_path:str, X_test_path:str, 
                   y_train_path:str, y_test_path:str) -> dict:
        """Evaluates a number of models using the same training and testing datasets.
        Args:
            models (dict): A dictionary of models to evaluate
            X_train_path (str): Path to training features
            y_train_path (str): Path to training labels
            X_test_path (str): Path to testing features
            y_test_path (str): Path to testing labels
            
            models = {"model_name" : model}
        Returns:
            pd.DataFrame: A dataframe of model names and their respective scores
            model: The best model
        """
        X_train = pd.read_csv(X_train_path).values
        y_train = pd.read_csv(y_train_path).values.ravel()
        X_test = pd.read_csv(X_test_path).values
        y_test = pd.read_csv(y_test_path).values.ravel()

        rf = RandomForestClassifier()
        xgb = XGBClassifier()
        dt = DecisionTreeClassifier()
        grad_boost = GradientBoostingClassifier()
        ada_boost = AdaBoostClassifier()
        et = ExtraTreeClassifier()

        models = {
            "Random Forest": rf,
            "XGBoost": xgb,
            "Decision Tree": dt,
            "Gradient Boosting": grad_boost,
            "Ada Boost": ada_boost,
            "Extra Tree": et
        }

        results = evaluate_models(models, X_train, y_train, X_test, y_test)

        os.makedirs(self.root_model_file, exist_ok=True)
        try:
            results.to_csv(self.model_results_file, index=False)
            logging.info("Model results saved successfully")
        except Exception as e:
            logging.error(f"Error saving model results - {e}")
            raise CustomException(e, sys)
        
        best_model = list(models.values())[results["Accuracy"].argmax()]
        best_model_name = results.sort_values("Accuracy", ascending=False).iloc[0]["Model"]
        try:
            logging.info(f"Saving the best model = {best_model_name}")    
            save_pickle(best_model, self.model_file)    
            logging.info("Best model saved successfully")
        except Exception as e:
            logging.error(f"Error saving the best model - {e}")
            raise CustomException(e, sys)
        
        return results, best_model
    

if __name__ == "__main__":
    model_training = ModelTraining()
    model_training.fit_models(X_train_path= "artifacts/transformed_data/train_features.csv", 
                              X_test_path= "artifacts/transformed_data/test_features.csv", 
                              y_train_path= "artifacts/transformed_data/train_labels.csv",
                              y_test_path= "artifacts/transformed_data/test_labels.csv")
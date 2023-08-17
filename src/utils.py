import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.logger import logging

def save_pickle(preprocessor, filename):
    """
    Save a scikit-learn object to a file using pickle.
    
    Parameters:
        preprocessor (scikit-learn transformer): The preprocessor object to be saved.
        filename (str): The name of the file to save the preprocessor object to.
    """
    try:
        with open(filename, 'wb') as f:
            pickle.dump(preprocessor, f)
        logging.info(f"Preprocessor saved to {filename}")
    except Exception as e:
        logging.error("Error saving preprocessor: %s", e)

def load_pickle(filename):
    """
    Load a scikit-learn preprocessor object from a file using pickle.
    
    Parameters:
        filename (str): The name of the file to load the preprocessor object from.
        
    Returns:
        preprocessor (scikit-learn transformer): The loaded preprocessor object.
    """
    try:
        with open(filename, 'rb') as f:
            preprocessor = pickle.load(f)
        logging.info(f"Preprocessor loaded from {filename}")
        return preprocessor
    except Exception as e:
        logging.error("Error loading preprocessor: %s", e)
        return None


def evaluate_models(models:dict, X_train:np.ndarray, y_train:np.ndarray, X_test:np.ndarray, y_test:np.ndarray) -> pd.DataFrame:
    """Evaluates a number of models using the same training and testing datasets.
    Args:
        models (dict): A dictionary of models to evaluate
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training labels
        X_test (np.ndarray): Testing features
        y_test (np.ndarray): Testing labels
    
        models = {"model_name" : model}
    Returns:
        pd.DataFrame: A dataframe of model names and their respective scores
    """
    accuracies = precisions = recalls = f1s = roc_aucs = np.zeros(len(models)) 

    for model_idx, (model_name, model) in enumerate(models.items()):
        logging.info(f"Evaluating {model_name}")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracies[model_idx] = accuracy_score(y_test, y_pred)
        precisions[model_idx] = precision_score(y_test, y_pred, average="weighted")
        recalls[model_idx] = recall_score(y_test, y_pred, average="weighted")
        f1s[model_idx] = f1_score(y_test, y_pred, average="weighted")
        logging.info(f"Score for {model_name} is {model.score(X_test, y_test)}")
        
    return pd.DataFrame({"Model": models.keys(), 
                         "Model_specs" : models.values(),
                         "Accuracy": accuracies, 
                         "Precision": precisions, 
                         "Recall": recalls, "F1": f1s})
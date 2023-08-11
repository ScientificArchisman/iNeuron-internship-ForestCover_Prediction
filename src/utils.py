import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from src.logger import logging

# Save an object as a pickle file
def save_pickle(obj, path):
    """Saves an object as a pickle file
    Args:
        obj (object): Object to be saved
        path (str): Path to save the object
    """
    try:
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        logging.info(f"Object saved at {path}")
    except Exception as e:
        logging.error(f"Error in saving the object at {path}")
        logging.error(e)


# Load a pickle file as an object
def load_pickle(path:str):
    """Loads a pickle file as an object
    Args:
        path (str): Path to load the object
    Returns:
        object: Object loaded from the pickle file
    """
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
        logging.info(f"Object loaded from {path}")
        return obj
    except Exception as e:
        logging.error(f"Error in loading the object at {path}")
        logging.error(e)
    

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
import pickle
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
    
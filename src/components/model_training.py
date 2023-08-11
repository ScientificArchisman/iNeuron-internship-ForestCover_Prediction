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
from src.utils import load_pickle, save_pickle
import optuna
from sklearn.metrics import accuracy_score
from optuna import Trial

# from resources.objective_func import objective




SEED = 123
def objective(trial : Trial):
    X_test = pd.read_csv("artifacts/transformed_data/test_features.csv")
    y_test = pd.read_csv("artifacts/transformed_data/test_labels.csv")
    classifier = trial.suggest_categorical('classifier', ['DecisionTree', 'ExtraTree', 'GradientBoosting', 'RandomForest', 'XGBoost'])

    if classifier == 'Decision Tree':
        params = {
            'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
            'splitter': trial.suggest_categorical('splitter', ['best', 'random']),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2', None]),
            'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', 0.0, 0.5),
            'min_impurity_split': trial.suggest_float('min_impurity_split', 0.0, 0.5),
            'random_state': SEED,  # For reproducibility
        }
        model = DecisionTreeClassifier(**params)

    elif classifier == 'Extra Tree':
        params = {
            'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
            'splitter': trial.suggest_categorical('splitter', ['best', 'random']),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2', None]),
            'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', 0.0, 0.5),
            'min_impurity_split': trial.suggest_float('min_impurity_split', 0.0, 0.5),
            'random_state': SEED,  # For reproducibility
        }
        model = ExtraTreeClassifier(**params)

    elif classifier == 'Gradient Boosting':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, step=0.05),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2']),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0, step=0.1),
            'random_state': SEED,  # For reproducibility
        }
        model = GradientBoostingClassifier(**params)

    elif classifier == 'Random Forest':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
            'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2']),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            'random_state': SEED,  # For reproducibility
        }
        model = RandomForestClassifier(**params)

    elif classifier == 'XGBoost':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, step=0.05),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3, 0.4, 0.5, 0.7]),
            'gamma': trial.suggest_categorical('gamma', [0.0, 0.1, 0.2, 0.3, 0.4]),
            'random_state': 42,  # For reproducibility
        }
        model = XGBClassifier(**params)

    # Using accuracy score to estimate model performance
    score = accuracy_score(y_test, model.predict(X_test).ravel())
    return score






class ModelTraining:
    def __init__(self, X_train_path:str, X_test_path:str, 
                   y_train_path:str, y_test_path:str):
        self.root_model_file = os.path.join("artifacts", "model_data")
        self.model_results_file = os.path.join(self.root_model_file, "model_results.csv")
        self.model_file = os.path.join(self.root_model_file, "model.pkl")
        self.tuned_model_file = os.path.join(self.root_model_file, "tuned_model.pkl")
        self.X_train = pd.read_csv(X_train_path).values
        self.y_train = pd.read_csv(y_train_path).values.ravel()
        self.X_test = pd.read_csv(X_test_path).values
        self.y_test = pd.read_csv(y_test_path).values.ravel()

    def fit_models(self) -> dict:
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
            best model: The best model
            best model name: The name of the best model
        """

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

        results = evaluate_models(models, self.X_train, self.y_train, self.X_test, self.y_test)

        os.makedirs(self.root_model_file, exist_ok=True)
        try:
            results.to_csv(self.model_results_file, index=False)
            logging.info("Model results saved successfully")
        except Exception as e:
            logging.error(f"Error saving model results - {e}")
            raise CustomException(e, sys)
        
        best_model = results.sort_values("Accuracy", ascending=False).iloc[0]["Model_specs"]
        best_model_name = results.sort_values("Accuracy", ascending=False).iloc[0]["Model"]
        try:
            logging.info(f"Saving the best model = {best_model_name}")    
            save_pickle(best_model, self.model_file)    
            logging.info("Best model saved successfully")
        except Exception as e:
            logging.error(f"Error saving the best model - {e}")
            raise CustomException(e, sys)
        
        return results, best_model, best_model_name
    

    def tune_model(self) -> dict:
        """Tunes a model using Optuna
        Args:
            None
        Returns:
            dict: A dictionary of the best parameters
            saves the best model"""


        # Define the objective function
        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=100, timeout=600, show_progress_bar=True)  # You can adjust the number of trials and timeout
    
        print("Number of finished trials: ", len(study.trials))
        print("Best trial:")
        trial = study.best_trial

        print("  Value: ", trial.value)
        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))

        # Save the best model
        # best_model = model(**trial.params)
        # best_model.fit(self.X_train, self.y_train)
        # save_pickle(best_model, self.tuned_model_file)
        # logging.info("Best model saved successfully")
        return trial.params


if __name__ == "__main__":
    model_training = ModelTraining(X_train_path= "artifacts/transformed_data/train_features.csv", 
                              X_test_path= "artifacts/transformed_data/test_features.csv", 
                              y_train_path= "artifacts/transformed_data/train_labels.csv",
                              y_test_path= "artifacts/transformed_data/test_labels.csv")
    # model_training.fit_models()
    model_training.tune_model()
    

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from xgboost import XGBClassifier
from src.utils import save_pickle, load_pickle
from src.logger import logging
from optuna import Trial
from sklearn.metrics import accuracy_score
import pandas as pd

SEED = 123
def objective(trial : Trial):
    X_test = pd.read_csv("artifacts/transformed_data/test_features.csv")
    y_test = pd.read_csv("artifacts/transformed_data/test_labels.csv")
    X_train = pd.read_csv("artifacts/transformed_data/train_features.csv")
    y_train = pd.read_csv("artifacts/transformed_data/train_labels.csv")
    classifier = trial.suggest_categorical('classifier', ['DecisionTree', 'ExtraTree', 'GradientBoosting', 'RandomForest', 'XGBoost'])

    if classifier == 'DecisionTree':
        params = {
            'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
            'splitter': trial.suggest_categorical('splitter', ['best', 'random']),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2', None]),
            'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', 0.0, 0.5),
            'min_impurity_split': trial.suggest_float('min_impurity_split', 0.0, 0.5),
            'random_state': 42,  # For reproducibility
        }

    elif classifier == 'ExtraTree':
        params = {
            'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
            'splitter': trial.suggest_categorical('splitter', ['best', 'random']),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2', None]),
            'min_impurity_decrease': trial.suggest_float('min_impurity_decrease', 0.0, 0.5),
            'min_impurity_split': trial.suggest_float('min_impurity_split', 0.0, 0.5),
            'random_state': 42,  # For reproducibility
        }

    elif classifier == 'GradientBoosting':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, step=0.05),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2']),
            'subsample': trial.suggest_float('subsample', 0.5, 1.0, step=0.1),
            'random_state': 42,  # For reproducibility
        }

    elif classifier == 'RandomForest':
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=100),
            'criterion': trial.suggest_categorical('criterion', ['gini', 'entropy']),
            'max_depth': trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features': trial.suggest_categorical('max_features', ['auto', 'sqrt', 'log2']),
            'bootstrap': trial.suggest_categorical('bootstrap', [True, False]),
            'random_state': 42,  # For reproducibility
        }

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

    # Using cross-validation to estimate model performance
    if classifier == 'DecisionTree':
        model = DecisionTreeClassifier(**params)
    elif classifier == 'ExtraTree':
        model = ExtraTreeClassifier(**params)
    elif classifier == 'GradientBoosting':
        model = GradientBoostingClassifier(**params)
    elif classifier == 'RandomForest':
        model = RandomForestClassifier(**params)
    elif classifier == 'XGBoost':
        model = XGBClassifier(**params)

    model.fit(X_train, y_train)

    # Using accuracy score to estimate model performance
    score = 0 #accuracy_score(y_test, model.predict(X_test).ravel())
    return score

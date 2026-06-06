# Library
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import numpy as np

seed = 42

# Model Baseline
def get_baseline_models():
    return {
        'Logistic Regression': LogisticRegression(
            max_iter=1000, random_state=seed
        ),
        'Decision Tree': DecisionTreeClassifier(
            random_state=seed
        ),
        'Random Forest': RandomForestClassifier(
            n_estimators=100, random_state=seed, n_jobs=-1
        ),
        'Extra Trees': ExtraTreesClassifier(
            n_estimators=100, random_state=seed, n_jobs=-1
        ),
        'XGBoost': XGBClassifier(
            random_state=seed, eval_metric='logloss', verbosity=0
        ),
        'LightGBM': LGBMClassifier(
            random_state=seed, verbose=-1
        ),
        'CatBoost': CatBoostClassifier(
            random_seed=seed, verbose=0
        ),
        'MLP': MLPClassifier(
            max_iter=500, random_state=seed
        ),
    }

# RandomizedSearchCV & HalvingRandomSearchCV
def get_param_distributions() :
    return {
        'Logistic Regression': {
            'C'      : np.logspace(-2, 1, 50),
            'solver' : ['lbfgs', 'saga'],
        },
        'Decision Tree': {
            'max_depth'        : list(range(3, 21)),
            'min_samples_split': list(range(2, 21)),
            'min_samples_leaf' : list(range(1, 11)),
            'criterion'        : ['gini', 'entropy'],
        },
        'Random Forest': {
            'n_estimators'     : list(range(50, 301, 50)),
            'max_depth'        : list(range(3, 21)),
            'min_samples_split': list(range(2, 21)),
            'min_samples_leaf' : list(range(1, 11)),
            'max_features'     : ['sqrt', 'log2'],
        },
        'Extra Trees': {
            'n_estimators'     : list(range(50, 301, 50)),
            'max_depth'        : list(range(3, 21)),
            'min_samples_split': list(range(2, 21)),
            'min_samples_leaf' : list(range(1, 11)),
            'max_features'     : ['sqrt', 'log2'],
        },
        'XGBoost': {
            'n_estimators'     : list(range(50, 301, 50)),
            'max_depth'        : list(range(3, 11)),
            'learning_rate'    : np.logspace(-2, np.log10(0.3), 30),
            'subsample'        : np.linspace(0.6, 1.0, 9),
            'colsample_bytree' : np.linspace(0.6, 1.0, 9),
            'reg_alpha'        : np.logspace(-8, 0, 20),
            'reg_lambda'       : np.logspace(-8, 0, 20),
        },
        'LightGBM': {
            'n_estimators'     : list(range(50, 301, 50)),
            'max_depth'        : list(range(3, 11)),
            'learning_rate'    : np.logspace(-2, np.log10(0.3), 30),
            'num_leaves'       : list(range(20, 151, 10)),
            'subsample'        : np.linspace(0.6, 1.0, 9),
            'colsample_bytree' : np.linspace(0.6, 1.0, 9),
            'reg_alpha'        : np.logspace(-8, 0, 20),
            'reg_lambda'       : np.logspace(-8, 0, 20),
        },
        'CatBoost': {
            'iterations'    : list(range(50, 301, 50)),
            'depth'         : list(range(3, 11)),
            'learning_rate' : np.logspace(-2, np.log10(0.3), 30),
            'l2_leaf_reg'   : np.logspace(-8, 1, 20),
            'border_count'  : list(range(32, 256, 16)),
        },
        'MLP': {
            'hidden_layer_sizes': [(64,), (128,), (64, 32), (128, 64), (128, 64, 32)],
            'activation'        : ['relu', 'tanh'],
            'alpha'             : np.logspace(-4, -1, 20),
            'learning_rate_init': np.logspace(-4, -1, 20),
        },
    }

# Optuna 
def suggest_params_optuna(trial, model_name: str, seed: int = seed):
    if model_name == 'Logistic Regression':
        params = {
            'C'        : trial.suggest_float('C', 0.01, 10.0, log=True),
            'solver'   : trial.suggest_categorical('solver', ['lbfgs', 'saga']),
            'max_iter' : 1000,
            'random_state': seed,
        }
        model = LogisticRegression(**params)

    elif model_name == 'Decision Tree':
        params = {
            'max_depth'        : trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf' : trial.suggest_int('min_samples_leaf', 1, 10),
            'criterion'        : trial.suggest_categorical('criterion', ['gini', 'entropy']),
            'random_state'     : seed,
        }
        model = DecisionTreeClassifier(**params)

    elif model_name == 'Random Forest':
        params = {
            'n_estimators'     : trial.suggest_int('n_estimators', 50, 300),
            'max_depth'        : trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf' : trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features'     : trial.suggest_categorical('max_features', ['sqrt', 'log2']),
            'random_state'     : seed,
            'n_jobs'           : -1,
        }
        model = RandomForestClassifier(**params)

    elif model_name == 'Extra Trees':
        params = {
            'n_estimators'     : trial.suggest_int('n_estimators', 50, 300),
            'max_depth'        : trial.suggest_int('max_depth', 3, 20),
            'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
            'min_samples_leaf' : trial.suggest_int('min_samples_leaf', 1, 10),
            'max_features'     : trial.suggest_categorical('max_features', ['sqrt', 'log2']),
            'random_state'     : seed,
            'n_jobs'           : -1,
        }
        model = ExtraTreesClassifier(**params)

    elif model_name == 'XGBoost':
        params = {
            'n_estimators'    : trial.suggest_int('n_estimators', 50, 300),
            'max_depth'       : trial.suggest_int('max_depth', 3, 10),
            'learning_rate'   : trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample'       : trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha'       : trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
            'reg_lambda'      : trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
            'random_state'    : seed,
            'eval_metric'     : 'logloss',
            'verbosity'       : 0,
        }
        model = XGBClassifier(**params)

    elif model_name == 'LightGBM':
        params = {
            'n_estimators'    : trial.suggest_int('n_estimators', 50, 300),
            'max_depth'       : trial.suggest_int('max_depth', 3, 10),
            'learning_rate'   : trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'num_leaves'      : trial.suggest_int('num_leaves', 20, 150),
            'subsample'       : trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha'       : trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
            'reg_lambda'      : trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
            'random_state'    : seed,
            'verbose'         : -1,
        }
        model = LGBMClassifier(**params)

    elif model_name == 'CatBoost':
        params = {
            'iterations'   : trial.suggest_int('iterations', 50, 300),
            'depth'        : trial.suggest_int('depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'l2_leaf_reg'  : trial.suggest_float('l2_leaf_reg', 1e-8, 10.0, log=True),
            'border_count' : trial.suggest_int('border_count', 32, 255),
            'random_seed'  : seed,
            'verbose'      : 0,
        }
        model = CatBoostClassifier(**params)

    elif model_name == 'MLP':
        n_layers = trial.suggest_int('n_layers', 1, 3)
        layers = tuple(
            trial.suggest_int(f'n_units_l{i}', 32, 256) for i in range(n_layers)
        )
        params = {
            'hidden_layer_sizes': layers,
            'activation'        : trial.suggest_categorical('activation', ['relu', 'tanh']),
            'alpha'             : trial.suggest_float('alpha', 1e-4, 0.1, log=True),
            'learning_rate_init': trial.suggest_float('learning_rate_init', 1e-4, 0.1, log=True),
            'max_iter'          : 500,
            'random_state'      : seed,
        }
        model = MLPClassifier(**params)

    else:
        raise ValueError(f"Model tidak dikenal: {model_name}")

    return model, params


def build_model_from_params(model_name: str, best_params: dict, seed: int = seed):
    if model_name == 'Logistic Regression':
        return LogisticRegression(**best_params, random_state=seed)
    elif model_name == 'Decision Tree':
        return DecisionTreeClassifier(**best_params, random_state=seed)
    elif model_name == 'Random Forest':
        return RandomForestClassifier(**best_params, random_state=seed, n_jobs=-1)
    elif model_name == 'Extra Trees':
        return ExtraTreesClassifier(**best_params, random_state=seed, n_jobs=-1)
    elif model_name == 'XGBoost':
        return XGBClassifier(**best_params, random_state=seed, eval_metric='logloss', verbosity=0)
    elif model_name == 'LightGBM':
        return LGBMClassifier(**best_params, random_state=seed, verbose=-1)
    elif model_name == 'CatBoost':
        return CatBoostClassifier(**best_params, random_seed=seed, verbose=0)
    elif model_name == 'MLP':
        return MLPClassifier(**best_params, random_state=seed)
    else:
        raise ValueError(f"Model tidak dikenal: {model_name}")

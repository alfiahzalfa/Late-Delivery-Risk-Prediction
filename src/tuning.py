# Library
import time
import numpy as np
import pandas as pd
import optuna

from sklearn.model_selection import StratifiedKFold, cross_val_score, RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv 
from sklearn.model_selection import HalvingRandomSearchCV
from sklearn.metrics import f1_score
from src.models import (
    suggest_params_optuna,
    get_param_distributions,
    build_model_from_params,
    get_baseline_models,
)
from src.evaluation import evaluate_model
from optuna.samplers import TPESampler
optuna.logging.set_verbosity(optuna.logging.WARNING)


# Konfigurasi keseluruhan
n_trials  = 50    # Optuna: jumlah trial per model
n_iter    = 50    # RandomizedSearch: jumlah iterasi
n_folds   = 5     # CV folds
metric   = 'f1'
seed      = 42

cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

model_names = [
    'Logistic Regression',
    'Decision Tree',
    'Random Forest',
    'Extra Trees',
    'XGBoost',
    'LightGBM',
    'CatBoost',
]


# Optuna
def _make_optuna_objective(model_name: str, X_train, y_train):
    def objective(trial):
        model, _ = suggest_params_optuna(trial, model_name, seed=seed)
        scores = cross_val_score(model, X_train, y_train,
                                 cv=cv, scoring=metric, n_jobs=-1)
        return scores.mean()
    return objective

## Tuning Satu Model Optuna TPE
def tune_optuna(model_name: str, X_train, X_test, y_train, y_test,
                label: str = '', n_trials: int = n_trials):
    print(f'\n{"="*60}')
    print(f'[Optuna] Tuning {model_name} — {label}')
    print(f'{"="*60}')

    study = optuna.create_study(direction='maximize', sampler=TPESampler(seed=seed))

    t0 = time.time()
    study.optimize(
        _make_optuna_objective(model_name, X_train, y_train),
        n_trials=n_trials,
        show_progress_bar=False,
    )
    elapsed = time.time() - t0

    best_params = study.best_params
    clean_params = {k: v for k, v in best_params.items()
                    if k not in ['random_state', 'random_seed', 'eval_metric',
                                 'verbose', 'verbosity', 'max_iter', 'n_jobs']}
    best_model = build_model_from_params(model_name, clean_params)
    best_model.fit(X_train, y_train)

    result = evaluate_model(model_name, best_model, X_train, X_test, y_train, y_test)
    result.update({
        '_tuning_method'   : 'Optuna',
        '_tuning_time_s'   : round(elapsed, 2),
        '_n_evaluations'   : n_trials,
        '_best_params'     : best_params,
        '_cv_best_score'   : round(study.best_value, 4),
        '_study'           : study,
    })

    _print_tuning_result(result)
    return result

## Tuning semua Optuna
def tune_all_optuna(X_train, X_test, y_train, y_test,
                    label: str = '', n_trials: int = n_trials):
    results = []
    for name in model_names:
        r = tune_optuna(name, X_train, X_test, y_train, y_test, label, n_trials)
        results.append(r)
    return results



# RANDOMIZEDSEARCHCV
def tune_randomized(model_name: str, X_train, X_test, y_train, y_test,
                    label: str = '', n_iter: int = n_iter):
    print(f'\n{"="*60}')
    print(f'[RandomizedSearchCV] Tuning {model_name} — {label}')
    print(f'{"="*60}')

    baseline_models = get_baseline_models()
    base_model = baseline_models[model_name]
    param_dists = get_param_distributions()

    if model_name not in param_dists:
        raise ValueError(f"Tidak ada param distribution untuk {model_name}")

    search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_dists[model_name],
        n_iter=n_iter,
        cv=cv,
        scoring=metric,
        n_jobs=-1,
        random_state=seed,
        refit=True,
    )

    t0 = time.time()
    search.fit(X_train, y_train)
    elapsed = time.time() - t0

    best_model = search.best_estimator_
    result = evaluate_model(model_name, best_model, X_train, X_test, y_train, y_test)
    result.update({
        '_tuning_method'  : 'RandomizedSearchCV',
        '_tuning_time_s'  : round(elapsed, 2),
        '_n_evaluations'  : n_iter * n_folds,
        '_best_params'    : search.best_params_,
        '_cv_best_score'  : round(search.best_score_, 4),
    })

    _print_tuning_result(result)
    return result

# Tuning semua Randomized
def tune_all_randomized(X_train, X_test, y_train, y_test,
                        label: str = '', n_iter: int = n_iter):
    results = []
    for name in model_names:
        r = tune_randomized(name, X_train, X_test, y_train, y_test, label, n_iter)
        results.append(r)
    return results


# HALVINGRANDOMSEARCHC
def tune_halving(model_name: str, X_train, X_test, y_train, y_test,
                 label: str = ''):    
    print(f'\n{"="*60}')
    print(f'[HalvingRandomSearchCV] Tuning {model_name} — {label}')
    print(f'{"="*60}')

    baseline_models = get_baseline_models()
    base_model = baseline_models[model_name]
    param_dists = get_param_distributions()

    if model_name not in param_dists:
        raise ValueError(f"Tidak ada param distribution untuk {model_name}")

    search = HalvingRandomSearchCV(
        estimator=base_model,
        param_distributions=param_dists[model_name],
        factor=3,
        cv=cv,
        scoring=metric,
        n_jobs=-1,
        random_state=seed,
        refit=True,
        min_resources='smallest',
        max_resources=500,
        aggressive_elimination=True,
    )

    t0 = time.time()
    search.fit(X_train, y_train)
    elapsed = time.time() - t0

    n_evals = sum(
        search.n_candidates_[i] * search.n_resources_[i] * n_folds
        for i in range(len(search.n_candidates_))
    ) if hasattr(search, 'n_candidates_') else -1

    best_model = search.best_estimator_
    result = evaluate_model(model_name, best_model, X_train, X_test, y_train, y_test)
    result.update({
        '_tuning_method'  : 'HalvingRandomSearchCV',
        '_tuning_time_s'  : round(elapsed, 2),
        '_n_evaluations'  : n_evals,
        '_best_params'    : search.best_params_,
        '_cv_best_score'  : round(search.best_score_, 4),
    })

    _print_tuning_result(result)
    return result


def tune_all_halving(X_train, X_test, y_train, y_test, label: str = ''):    
    results = []
    for name in model_names:
        r = tune_halving(name, X_train, X_test, y_train, y_test, label)
        results.append(r)
    return results


# Result
def _print_tuning_result(result: dict):
    print(f"  Method   : {result['_tuning_method']}")
    print(f"  Time     : {result['_tuning_time_s']}s")
    print(f"  CV F1    : {result.get('_cv_best_score', '-')}")
    print(f"  Test F1  : {result['F1-Score']}")
    print(f"  ROC-AUC  : {result['ROC-AUC']}")
    print(f"  Params   : {result['_best_params']}")

# Tabel Perbandingan
def summarize_tuning_comparison(results_baseline: list,
                                 results_optuna: list,
                                 results_random: list,
                                 results_halving: list,
                                 label: str = '') -> pd.DataFrame:

    rows = []
    keys = ['Model', 'F1-Score', 'ROC-AUC', 'Tuning Time (s)', 'N Evaluations', 'Method']

    for r in results_baseline:
        rows.append({
            'Model'         : r['Model'],
            'Method'        : 'Baseline',
            'F1-Score'      : r['F1-Score'],
            'ROC-AUC'       : r['ROC-AUC'],
            'Tuning Time(s)': r.get('Train Time (s)', 0),
            'N Evaluations' : 1,
        })
    for group, method_name in [
        (results_optuna,  'Optuna'),
        (results_random,  'RandomizedSearchCV'),
        (results_halving, 'HalvingRandomSearchCV'),
    ]:
        for r in group:
            rows.append({
                'Model'         : r['Model'],
                'Method'        : method_name,
                'F1-Score'      : r['F1-Score'],
                'ROC-AUC'       : r['ROC-AUC'],
                'Tuning Time(s)': r.get('_tuning_time_s', 0),
                'N Evaluations' : r.get('_n_evaluations', -1),
            })

    df = pd.DataFrame(rows)
    print(f'\n=== Perbandingan 3 Metode Tuning [{label}] ===')
    print(df.pivot_table(
        index='Model', columns='Method', values='F1-Score'
    ).round(4).to_string())
    return df

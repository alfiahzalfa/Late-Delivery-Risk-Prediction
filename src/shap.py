# Library
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap

# Pilih explainer berdasarkan model
def get_shap_explainer(model, X_train: pd.DataFrame):
    model_class = type(model).__name__

    tree_models = [
        'RandomForestClassifier', 'ExtraTreesClassifier',
        'DecisionTreeClassifier', 'XGBClassifier',
        'LGBMClassifier', 'CatBoostClassifier',
    ]

    if model_class in tree_models:
        explainer = shap.TreeExplainer(model)
    elif model_class == 'LogisticRegression':
        explainer = shap.LinearExplainer(model, X_train)
    else:
        # MLP atau model lain, pakai subset 200 sampel agar tidak OOM
        sample = shap.sample(X_train, 200, random_state=42)
        explainer = shap.KernelExplainer(model.predict_proba, sample)

    return explainer

# Hitung SHAP values
def compute_shap_values(explainer, X: pd.DataFrame):
    shap_values = explainer.shap_values(X)
    # Untuk binary classification beberapa explainer return list [class0, class1]
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    return shap_values

# Visualisasi plot summary SHAP values
def plot_summary(shap_values, X: pd.DataFrame, label: str, max_display: int = 20):
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, max_display=max_display, show=False)
    plt.title(f'SHAP Summary — {label}\n', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()

# Visualisasi plot bar SHAP values
def plot_bar(shap_values, X: pd.DataFrame, label: str, max_display: int = 20):
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values, X, plot_type='bar',
                      max_display=max_display, show=False)
    plt.title(f'SHAP Feature Importance — {label}\n', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()

# Visualisasi plot waterfall SHAP values
def plot_waterfall(explainer, X: pd.DataFrame, label: str, idx: int = 0):
    sv = explainer.shap_values(X.iloc[[idx]])

    # kalau list [class0, class1] → ambil class 1
    if isinstance(sv, list):
        sv = sv[1]

    # squeeze sampai 1D — ulangi sampai shape (n_features,)
    while hasattr(sv, 'ndim') and sv.ndim > 1:
        sv = sv[0]

    # expected value → ambil class 1 kalau list/array
    base = explainer.expected_value
    if isinstance(base, (list, np.ndarray)):
        base = base[1]

    explanation = shap.Explanation(
        values        = sv,
        base_values   = float(base),
        data          = X.iloc[idx].values,
        feature_names = X.columns.tolist(),
    )

    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(explanation, show=False)
    plt.title(f'SHAP Waterfall — {label} (sampel #{idx})\n',
              fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    shap.plots.waterfall(explanation, show=False)
    plt.title(f'SHAP Waterfall — {label} (sampel #{idx})\n',
              fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.show()

# Visualisasi plot dependence SHAP values
def plot_dependence(shap_values, X: pd.DataFrame, feature: str,
                    label: str, interaction_feature: str = None):
    plt.figure(figsize=(8, 5))
    shap.dependence_plot(
        feature, shap_values, X,
        interaction_index=interaction_feature,
        show=False
    )
    plt.title(f'SHAP Dependence — {feature} [{label}]\n',
              fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.show()

# Mengembalikan DataFrame top-N fitur berdasarkan mean |SHAP|
def get_top_features(shap_values, X: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    sv = shap_values
    # squeeze sampai 2D (n_samples, n_features)
    while sv.ndim > 2:
        sv = sv[0] if sv.shape[0] == 1 else sv[:, :, 1] if sv.shape[-1] == 2 else sv[0]

    importance = pd.DataFrame({
        'Feature'    : X.columns,
        'Mean |SHAP|': np.abs(sv).mean(axis=0),
    }).sort_values('Mean |SHAP|', ascending=False).reset_index(drop=True)
    importance.index += 1
    print(f'\nTop-{n} Fitur berdasarkan SHAP:')
    print(importance.head(n).to_string())
    return importance

# Pipeline lengkap SHAP analysis
def run_shap_analysis(model, X_train: pd.DataFrame, X_test: pd.DataFrame,
                      label: str, n_sample: int = 500):
    print(f'\n{"="*55}')
    print(f'SHAP Analysis — {label}')
    print(f'{"="*55}')

    # Subset agar cepat (khususnya KernelExplainer)
    X_sample = X_test.iloc[:n_sample] if len(X_test) > n_sample else X_test

    explainer   = get_shap_explainer(model, X_train)
    shap_values = compute_shap_values(explainer, X_sample)

    plot_summary(shap_values, X_sample, label)
    plot_bar(shap_values, X_sample, label)
    plot_waterfall(explainer, X_sample, label, idx=0)

    top_features = get_top_features(shap_values, X_sample)
    return shap_values, top_features

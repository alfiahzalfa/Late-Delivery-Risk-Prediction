# Library
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    roc_auc_score, 
    confusion_matrix, 
    ConfusionMatrixDisplay,
    classification_report, 
    roc_curve,
)


# Train Model 
def evaluate_model(name: str, model, X_train, X_test, y_train, y_test) -> dict:
    t0 = time.time()
    model.fit(X_train, y_train)
    elapsed = round(time.time() - t0, 2)

    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    return {
        'Model'          : name,
        'Accuracy'       : round(accuracy_score(y_test, y_pred), 4),
        'Precision'      : round(precision_score(y_test, y_pred), 4),
        'Recall'         : round(recall_score(y_test, y_pred), 4),
        'F1-Score'       : round(f1_score(y_test, y_pred), 4),
        'ROC-AUC'        : round(roc_auc_score(y_test, y_proba), 4),
        'Train Time (s)' : elapsed,
        '_model'         : model,
        '_y_pred'        : y_pred,
        '_y_proba'       : y_proba,
    }

# Train and Evaluate
def run_all_models(models: dict, X_train, X_test, y_train, y_test, label: str) -> list:
    print(f'\n{"="*60}')
    print(f'MODELLING — {label}')
    print(f'Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,} | Fitur: {X_train.shape[1]}')
    print(f'{"="*60}')
    results = []
    for name, model in models.items():
        print(f'  Training {name}...', end=' ', flush=True)
        r = evaluate_model(name, model, X_train, X_test, y_train, y_test)
        results.append(r)
        print(f'F1={r["F1-Score"]} | AUC={r["ROC-AUC"]} ({r["Train Time (s)"]}s)')
    return results


# Tabel Evaluate
def show_comparison_table(results: list, label: str) -> pd.DataFrame:
    cols = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC', 'Train Time (s)']
    df = pd.DataFrame([{k: r[k] for k in cols} for r in results])
    df = df.sort_values('F1-Score', ascending=False).reset_index(drop=True)
    df.index += 1
    print(f'\n--- Tabel Perbandingan [{label}] ---')
    print(df.to_string())
    print(f'\nModel terbaik: {df.iloc[0]["Model"]} (F1={df.iloc[0]["F1-Score"]})')
    return df


# Visualisasi hasil Model
def plot_comparison(df_res: pd.DataFrame, label: str, save_path: str = None):
    metrics      = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    models_name  = df_res['Model'].tolist()
    x            = np.arange(len(models_name))
    width        = 0.15
    colors       = ['#3A7EBF', '#2EAF7D', '#E05A3A', '#7C6FCD', '#F5A623']

    fig, ax = plt.subplots(figsize=(15, 6))
    fig.patch.set_facecolor('#FAFAFA')

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        vals = df_res[metric].tolist()
        bars = ax.bar(x + i * width, vals, width, label=metric,
                      color=color, alpha=0.85, edgecolor='white')
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=6, rotation=90)

    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(models_name, fontsize=9)
    ax.set_ylim(0, 1.18)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title(f'Perbandingan Model — {label}\n', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9, loc='lower right')
    ax.spines[['top', 'right']].set_visible(False)
    ax.set_facecolor('#FAFAFA')
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()

# Visualisasi Confusion Matrix
def plot_confusion_matrices(results: list, y_test, label: str):
    n      = len(results)
    n_cols = 4
    n_rows = (n + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5 * n_rows))
    fig.patch.set_facecolor('#FAFAFA')
    axes = axes.flatten()
    for i, res in enumerate(results):
        cm   = confusion_matrix(y_test, res['_y_pred'])
        disp = ConfusionMatrixDisplay(cm, display_labels=['Tidak Terlambat', 'Terlambat'])
        disp.plot(ax=axes[i], colorbar=False, cmap='Blues')
        axes[i].set_title(res['Model'], fontsize=9, fontweight='bold')
    for j in range(n, len(axes)):
        axes[j].set_visible(False)
    fig.suptitle(f'Confusion Matrix — {label}\n', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()

# Visualisasi ROC Curve
def plot_roc_curves(results: list, y_test, label: str):
    palette = ['#3A7EBF', '#2EAF7D', '#E05A3A', '#7C6FCD', '#F5A623',
               '#333333', '#E91E8C', '#00BCD4']
    fig, ax = plt.subplots(figsize=(9, 7))
    fig.patch.set_facecolor('#FAFAFA')
    for res, color in zip(results, palette):
        fpr, tpr, _ = roc_curve(y_test, res['_y_proba'])
        ax.plot(fpr, tpr, color=color, linewidth=2,
                label=f'{res["Model"]} (AUC={res["ROC-AUC"]})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random')
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title(f'ROC Curve — {label}\n', fontsize=13, fontweight='bold')
    ax.legend(fontsize=8, loc='lower right')
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()

# Visualisasi Perbandingan Sebelum dan  Sesudah Tuning
def plot_before_after(before: dict, after: dict, label: str):
    metrics      = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
    before_vals  = [before[m] for m in metrics]
    after_vals   = [after[m]  for m in metrics]
    x = np.arange(len(metrics))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    fig.patch.set_facecolor('#FAFAFA')
    ax.bar(x - w / 2, before_vals, w, label='Sebelum Tuning', color='#AAAAAA', alpha=0.85, edgecolor='white')
    bars2 = ax.bar(x + w / 2, after_vals, w, label='Setelah Tuning', color='#2EAF7D', alpha=0.85, edgecolor='white')

    for bar, bv, av in zip(bars2, before_vals, after_vals):
        diff   = av - bv
        symbol = '+' if diff >= 0 else ''
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f'{symbol}{diff:.4f}', ha='center', va='bottom', fontsize=8,
                color='green' if diff >= 0 else 'red')

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylim(min(before_vals + after_vals) - 0.05, 1.12)
    ax.set_ylabel('Score', fontsize=11)
    ax.set_title(f'Sebelum vs Sesudah Tuning — {label}\n', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.show()

# Visualisasi Perbandingan 3 Metode Tuning
def plot_tuning_comparison(df_comparison: pd.DataFrame, label: str):
    methods = ['Baseline', 'Optuna', 'RandomizedSearchCV', 'HalvingRandomSearchCV']
    colors  = {'Baseline': '#AAAAAA', 'Optuna': '#3A7EBF',
                'RandomizedSearchCV': '#E05A3A', 'HalvingRandomSearchCV': '#2EAF7D'}

    models  = df_comparison['Model'].unique().tolist()
    x       = np.arange(len(models))
    w       = 0.18

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.patch.set_facecolor('#FAFAFA')

    # F1-Score
    ax = axes[0]
    for i, method in enumerate(methods):
        sub  = df_comparison[df_comparison['Method'] == method].set_index('Model')
        vals = [sub.loc[m, 'F1-Score'] if m in sub.index else 0 for m in models]
        ax.bar(x + i * w, vals, w, label=method, color=colors[method], alpha=0.85, edgecolor='white')
    ax.set_xticks(x + w * 1.5)
    ax.set_xticklabels(models, rotation=30, ha='right', fontsize=8)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel('F1-Score')
    ax.set_title('F1-Score per Metode Tuning')
    ax.legend(fontsize=7)
    ax.grid(axis='y', alpha=0.3)

    # Waktu Tuning
    ax = axes[1]
    for i, method in enumerate([m for m in methods if m != 'Baseline']):
        sub  = df_comparison[df_comparison['Method'] == method].set_index('Model')
        vals = [sub.loc[m, 'Tuning Time(s)'] if m in sub.index else 0 for m in models]
        ax.bar(x + i * w, vals, w, label=method, color=colors[method], alpha=0.85, edgecolor='white')
    ax.set_xticks(x + w)
    ax.set_xticklabels(models, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('Waktu (detik)')
    ax.set_title('Waktu Tuning per Metode')
    ax.legend(fontsize=7)
    ax.grid(axis='y', alpha=0.3)

    # Jumlah Evaluasi
    ax = axes[2]
    for i, method in enumerate([m for m in methods if m != 'Baseline']):
        sub  = df_comparison[df_comparison['Method'] == method].set_index('Model')
        vals = [sub.loc[m, 'N Evaluations'] if m in sub.index else 0 for m in models]
        ax.bar(x + i * w, vals, w, label=method, color=colors[method], alpha=0.85, edgecolor='white')
    ax.set_xticks(x + w)
    ax.set_xticklabels(models, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('Jumlah Evaluasi')
    ax.set_title('Jumlah Evaluasi per Metode')
    ax.legend(fontsize=7)
    ax.grid(axis='y', alpha=0.3)

    fig.suptitle(f'Perbandingan Metode Tuning — {label}\n', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()


def save_model(model, path: str):
    joblib.dump(model, path)
    print(f"Model disimpan: {path}")

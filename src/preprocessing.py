# Library
import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns 

from scipy.stats import chi2_contingency


# Load Data
def load_data(filepath: str, encoding: str = 'latin1'):
    df = pd.read_csv(filepath, encoding=encoding)
    print(f'Dataset berjumlah {df.shape[0]} baris dan {df.shape[1]} kolom')
    return df

# Info Data
def check_info(df: pd.DataFrame):
    print(df.info())
    print('\n')
    return df

def check_describe(df: pd.DataFrame):
    print(df.describe())
    print('\n')
    return df

# Drop kolom yang tidak relevan
def drop_irrelevant_columns(df: pd.DataFrame):
    # data leakage (data yang ada setelah proses pengiriman)
    leakage = [
        'Delivery Status',              # hasil status pengiriman
        'shipping date (DateOrders)',   # tanggal pengiriman
    ]
    # data personal
    personal_identifier = [
        'Customer Email',
        'Customer Fname',
        'Customer Lname',
        'Customer Password',
        'Customer Street',
    ]
    # data yang mengandung ID
    id = [
        'Customer Id', 
        'Order Customer Id', 
        'Order Id',
        'Order Item Id', 
        'Order Item Cardprod Id',
        'Product Card Id', 
        'Product Category Id',
        'Category Id', 
        'Department Id',
    ]
    # data yang tidak digunakan dalam memprediksi telat pengiriman
    other = [
        'Product Image',
        'Customer Zipcode',
        'Latitude',
        'Longitude',
        'Product Description',
        'Order Zipcode',    
    ]
    
    # drop kolom
    drop_cols = leakage + personal_identifier + id + other
    drop_existing = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=drop_existing)
    print(f"Kolom yang telah dihapus ({len(drop_existing)}): {drop_existing}")
    return df

# Missing Value
def handle_missing_values(df: pd.DataFrame):
    # cek missing value
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Values': missing,
        'Percentage (%)': missing_pct
    })
    missing_df = missing_df[missing_df['Missing Values'] > 0].sort_values('Percentage (%)', ascending=False)
    if not missing_df.empty:
        print("Missing Values:")
        print(missing_df)
    else:
        print("Tidak ada missing values.")
    # Drop kolom yang missing value-nya lebih dari 50%
    cols_to_drop = missing_pct[missing_pct > 50].index.tolist()
    before = df.shape[1]
    df = df.drop(columns=cols_to_drop)
    after = df.shape[1]
    print(f"\nKolom yang memiliki missing value lebih dari 50%: {cols_to_drop}")
    print(f"Jumlah kolom sebelum didrop: {before}")
    print(f"Jumlah kolom setelah didrop: {after}")
    return df

# Duplicates
def handle_duplicate_rows(df: pd.DataFrame):
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        df = df.drop_duplicates()
        print(f"Total duplicates : {duplicate_count}")
        print(f"Jumlah baris setelah didrop: {len(df)}")
    else:
        print("Tidak ada data yang duplicates.")
    return df

# IQR
def iqr_capping(df: pd.DataFrame, cols=None, verbose: bool=True):
    df_capped = df.copy()
    num_cols = cols if cols is not None else df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    if verbose:
        print('### IQR Capping (Winsorization) ###')
        print(f'{"Kolom":<35} {"Before":>10} {"After":>10}')
        print('-' * 60)
    for col in num_cols:
        # Hitung IQR
        Q1 = df_capped[col].quantile(0.25)
        Q3 = df_capped[col].quantile(0.75)
        IQR = Q3 - Q1
        # tentukan batas atas dan bawah
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        # outlier sebelum capping
        before = ((df_capped[col] < lower) | (df_capped[col] > upper)).sum()
        # capping outlier
        df_capped[col] = df_capped[col].clip(lower=lower, upper=upper)
        after = ((df_capped[col] < lower) | (df_capped[col] > upper)).sum()
        if verbose:
            print(f'{col:<35} {before:>10,} {after:>10,}')
    return df_capped

# Hapus fitur redundant dari analisis multikolinearitas
def drop_redundant_features(df: pd.DataFrame):
    redundant = [
        'Order Profit Per Order',   # duplicate dari Benefit per order
        'Order Item Product Price', # duplicate dari Product Price
        'Order Item Total',         # duplicate dari Sales dan Sales per customer
    ]
    redundant_existing = [c for c in redundant if c in df.columns]
    df = df.drop(columns=redundant_existing)
    print(f"Fitur redundan yang didrop: {redundant_existing}")
    return df

# Chi-Square Test fitur kategorik vs target
def run_chi2_test(df: pd.DataFrame, target: str = 'Late_delivery_risk') -> list:
    df_cat = df.select_dtypes(include='object')
    print(f'\n{"Fitur Kategorik":<40} {"Chi2":>12} {"p-value":>12} {"Signifikan":>12}')
    print('-' * 80)
    sig_cat = []
    for col in df_cat.columns:
        ct = pd.crosstab(df[col], df[target])
        chi2, p, dof, _ = chi2_contingency(ct)
        sig = 'Ya' if p < 0.05 else 'Tidak'
        if sig == 'Ya':
            sig_cat.append(col)
        print(f'{col:<40} {chi2:>12.2f} {p:>12.4f} {sig:>12}')
    print(f'\nFitur yang signifikan ({len(sig_cat)}): {sig_cat}')
    return sig_cat

# Hapus fitur kategorik yang tidak signifikan atau high cardinality
def drop_low_relevance_categoricals(df: pd.DataFrame):
    drop_cat = [
        # p-value > 0.05
        'Category Name', 
        'Customer Country', 
        'Customer Segment',
        'Department Name', 
        'Market', 
        'Product Name'
        # high cardinality
        'Customer City', 
        'Order City', 
        'Order State',
        # data leakage
        'order date (DateOrders)', 
        'Order Status',
        # redudant
        'Order Country',
    ]   
    drop_existing = [c for c in drop_cat if c in df.columns]
    df = df.drop(columns=drop_existing)
    print(f"Kolom kategorik yang didrop ({len(drop_existing)}): {drop_existing}")
    return df

# Pipeline preprocessing 
def full_preprocessing_pipeline(filepath: str, target: str = 'Late_delivery_risk') -> pd.DataFrame:
    df = load_data(filepath)
    df = drop_irrelevant_columns(df)
    df = handle_missing_values(df)
    df = handle_duplicate_rows(df)
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    df = drop_redundant_features(df)
    df = drop_low_relevance_categoricals(df)

    # Drop kolom konstan (korelasi NaN terhadap target)
    df_num = df.select_dtypes(include=['float64', 'int64'])
    num_cols_clean = [c for c in df_num.columns if c != target]
    corr_vs_target = df[num_cols_clean].corrwith(df[target]).abs().dropna()
    nan_corr_cols = [c for c in num_cols_clean if c not in corr_vs_target.index]
    if nan_corr_cols:
        df = df.drop(columns=nan_corr_cols)
        print(f'Kolom konstan di-drop: {nan_corr_cols}')

    print(f'\nDataset setelah preprocessing: {df.shape[0]:,} baris, {df.shape[1]} kolom')
    return df
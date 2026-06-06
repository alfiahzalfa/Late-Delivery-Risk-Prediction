# Library
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Target yang digunakan
Target = 'Late_delivery_risk'

# Fitur pada Skenario 1 : Pre-Shipment (info minimal saat order masuk)
features_1 = [
    'Days for shipment (scheduled)',
    'Shipping Mode',
    'Type',
    'Order Item Quantity',
    'Product Price',
    'Order Item Discount Rate',
]

features_2 = [
    'Days for shipment (scheduled)',
    'Shipping Mode',
    'Type',
    'Order Item Quantity',
    'Order Item Discount Rate',
    'Product Price',
    'Order Item Profit Ratio',
    'discount_per_unit',
    'Order Region',
    'Customer State',
]

shipping_order = {
    'Same Day'      : 0,
    'First Class'   : 1,
    'Second Class'  : 2,
    'Standard Class': 3,
}

# tambah fitur discount per unit
def add_engineered_features(df: pd.DataFrame):
    df = df.copy()
    df['discount_per_unit'] = df['Order Item Discount Rate'] * df['Product Price']
    print("Fitur baru ditambahkan: discount_per_unit")
    return df

# Encode
def encode_scenario(df, features, target='Late_delivery_risk'):
    df_enc = df[features + [target]].copy()
    
    if 'Shipping Mode' in df_enc.columns:
        df_enc['Shipping Mode'] = df_enc['Shipping Mode'].map(shipping_order)
    
    ohe_cols = [c for c in ['Type', 'Order Status', 'Category Name', 'Department Name', 'Order Region', 'Customer State']
                if c in df_enc.columns]
    
    print("OHE cols:", ohe_cols)                          
    print("Dtypes sebelum OHE:", df_enc.dtypes[ohe_cols]) 
    
    if ohe_cols:
        df_enc = pd.get_dummies(df_enc, columns=ohe_cols, drop_first=True, dtype=int)
    
    return df_enc

# hasil encode 2 skenario
def prepare_scenarios(df: pd.DataFrame, target: str = Target):
    df = add_engineered_features(df)
    df_s1 = encode_scenario(df, features_1, target)
    df_s2 = encode_scenario(df, features_2, target)
    print(f"Skenario 1 setelah encoding: {df_s1.shape}")
    print(f"Skenario 2 setelah encoding: {df_s2.shape}")
    return df_s1, df_s2

# Train test Split
def split_and_scale(df_enc: pd.DataFrame,
                    target: str = Target,
                    test_size: float = 0.2,
                    random_state: int = 42,
                    use_scaling: bool = False,
                    scaler_path: str = None):

    X = df_enc.drop(columns=target).copy()
    y = df_enc[target].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = None
    if use_scaling:
        num_cols = X_train.select_dtypes(include='number').columns.tolist()
        scaler = StandardScaler()
        X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
        X_test[num_cols]  = scaler.transform(X_test[num_cols])
        if scaler_path:
            joblib.dump(scaler, scaler_path)
            print(f"Scaler disimpan: {scaler_path}")

    print(f"Train: {X_train.shape[0]:,} | Test: {X_test.shape[0]:,} | Fitur: {X_train.shape[1]}")
    return X_train, X_test, y_train, y_test, scaler

# Pipeline features
def prepare_all(df: pd.DataFrame,
                target: str = Target,
                scaler_s1_path: str = 'models/scaler_s1.pkl',
                scaler_s2_path: str = 'models/scaler_s2.pkl'):
    df_s1, df_s2 = prepare_scenarios(df, target)

    print("\n--- Skenario 1 (Pre-Shipment) ---")
    X_train_s1, X_test_s1, y_train_s1, y_test_s1, scaler_s1 = split_and_scale(
        df_s1, target, use_scaling=False 
    )

    print("\n--- Skenario 2 (Full Order Profile) ---")
    X_train_s2, X_test_s2, y_train_s2, y_test_s2, scaler_s2 = split_and_scale(
        df_s2, target, use_scaling=False  
    )

    return {
        's1': {
            'X_train': X_train_s1, 'X_test': X_test_s1,
            'y_train': y_train_s1, 'y_test': y_test_s1,
            'scaler': scaler_s1, 
        },
        's2': {
            'X_train': X_train_s2, 'X_test': X_test_s2,
            'y_train': y_train_s2, 'y_test': y_test_s2,
            'scaler': scaler_s2,  
        },
    }

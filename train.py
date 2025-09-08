# train.py
import os
import sys
import joblib
import numpy as np
import pandas as pd
import kagglehub
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Config - you can edit
KAGGLE_DATASET = "alifarahmandfar/continuous-groundwater-level-measurements-2023"
PREFERRED_CSV = "gwl-daily.csv"
SAMPLE_MAX_ROWS = 200_000
RANDOM_STATE = 42
MODEL_OUTPUT = os.environ.get("MODEL_PATH", "./models/model.joblib")

os.makedirs(os.path.dirname(MODEL_OUTPUT) or ".", exist_ok=True)

def find_target_column(df):
    candidates = ['lev_va','gwl','groundwater_level_m','groundwater_level','value','wl','water_level','level','depth']
    col_map = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in col_map:
            return col_map[cand]
    for c in df.select_dtypes(include=[np.number]).columns:
        if any(k in c.lower() for k in ['lev','gwl','water','level','wl','depth','value']):
            return c
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    return numeric_cols[0] if numeric_cols else None

def simple_preprocess(df, target_col):
    df = df[df[target_col].notna()].copy()
    drop_names = {'site_no','site_id','station_id','station_no','station','id','lev_dt','date','datetime','timestamp','time','lev_date','date_time'}
    drop_cols = [c for c in df.columns if c.lower() in drop_names]
    df = df.drop(columns=drop_cols, errors='ignore')
    y = df[target_col].astype(float)
    X = df.drop(columns=[target_col], errors='ignore')

    one_hot_cols = []
    drop_high_card = []
    for col in X.columns:
        if pd.api.types.is_numeric_dtype(X[col]):
            continue
        conv = pd.to_numeric(X[col], errors='coerce')
        non_na_ratio = conv.notna().mean()
        uniques = X[col].nunique(dropna=True)
        if non_na_ratio > 0.9 and uniques > 1:
            X[col] = conv
            continue
        if uniques <= 50:
            one_hot_cols.append(col)
        else:
            drop_high_card.append(col)

    X = X.drop(columns=drop_high_card, errors='ignore')
    if one_hot_cols:
        X = pd.get_dummies(X, columns=one_hot_cols, drop_first=True)
    X = X.select_dtypes(include=[np.number])
    if X.isna().any().any():
        medians = X.median()
        X = X.fillna(medians)
    return X, y

def main():
    print("Downloading dataset via kagglehub...")
    path = kagglehub.dataset_download(KAGGLE_DATASET)
    print("Dataset path:", path)
    files = os.listdir(path)
    csv_files = [f for f in files if f.lower().endswith(".csv")]
    if not csv_files:
        print("No CSV files found.")
        sys.exit(1)
    csv_file = os.path.join(path, PREFERRED_CSV) if PREFERRED_CSV in csv_files else os.path.join(path, csv_files[0])
    print("Loading:", csv_file)
    df = pd.read_csv(csv_file, low_memory=False)
    print("Loaded rows:", df.shape[0], "cols:", df.shape[1])
    target_col = find_target_column(df)
    if target_col is None:
        print("Could not find a numeric target.")
        sys.exit(1)
    print("Target column:", target_col)
    X, y = simple_preprocess(df, target_col)
    if X.shape[0] > SAMPLE_MAX_ROWS:
        sampled_index = X.sample(n=SAMPLE_MAX_ROWS, random_state=RANDOM_STATE).index
        X = X.loc[sampled_index].reset_index(drop=True)
        y = y.loc[sampled_index].reset_index(drop=True)
    else:
        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

    x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=RANDOM_STATE, test_size=0.2)

    model = RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state=RANDOM_STATE)
    print("Training model...")
    model.fit(x_train, y_train)
    print("Saving model to:", MODEL_OUTPUT)
    joblib.dump({'model': model, 'features': list(X.columns)}, MODEL_OUTPUT)

    y_test_pred = model.predict(x_test)
    test_r2 = r2_score(y_test, y_test_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_mae = mean_absolute_error(y_test, y_test_pred)
    print(f"Test R²: {test_r2:.4f}, RMSE: {np.sqrt(test_mse):.2f}, MAE: {test_mae:.2f}")

if __name__ == "__main__":
    main()

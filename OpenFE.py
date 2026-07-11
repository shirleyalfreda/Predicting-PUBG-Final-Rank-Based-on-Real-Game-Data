import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import ExtraTreesRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.feature_selection import VarianceThreshold, mutual_info_regression
from scipy.stats import zscore
from openfe import OpenFE

def add_features(df):
    df = df.copy()
    df["total_dist"] = df["player_dist_ride"] + df["player_dist_walk"]
    df["kd_ratio"] = df["player_kills"] / (1 + df["player_dbno"])
    df.loc[np.isinf(df["kd_ratio"]), "kd_ratio"] = np.nan
    df["kd_ratio"] = df["kd_ratio"].fillna(df["kd_ratio"].mean())
    return df

def safe_openfe_fit_transform(openfe, X, y, X_test=None):
    try:
        openfe.fit(X, y)
        train_fe = openfe.transform(X).fillna(0)
        test_fe = openfe.transform(X_test).fillna(0) if X_test is not None else None
        return train_fe, test_fe
    except Exception as e:
        print(f"失败: {str(e)}")
        return None, None

def main():
    train_path = r"C:\Users\banana\Desktop\privacy\school\data mining\数据挖掘期末\pubg_train.csv\pubg_train.csv"
    test_path = r"C:\Users\banana\Desktop\privacy\school\data mining\数据挖掘期末\pubg_test.csv\pubg_test.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    y = train_df["team_placement"].copy()
    train_df = train_df.drop(columns=["team_placement"])

    for col in train_df.columns:
        if train_df[col].dtype in [np.float64, np.int64]:
            train_df[col] = train_df[col].fillna(train_df[col].mean())
            test_df[col] = test_df[col].fillna(test_df[col].mean())
        else:
            train_df[col] = train_df[col].fillna("missing")
            test_df[col] = test_df[col].fillna("missing")

    numeric_cols = train_df.select_dtypes(include=np.number).columns
    z_scores = zscore(train_df[numeric_cols])
    mask = (np.abs(z_scores) < 3).all(axis=1)
    train_df = train_df[mask]
    y = y[mask]

    train_df = add_features(train_df)
    test_df = add_features(test_df)

    num_features = train_df.select_dtypes(include=np.number).columns.tolist()
    cat_features = train_df.select_dtypes(exclude=np.number).columns.tolist()

    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), num_features),
        ("cat", OneHotEncoder(handle_unknown='ignore'), cat_features)
    ])

    X_train = preprocessor.fit_transform(train_df)
    X_test = preprocessor.transform(test_df)

    print("\n=== OpenFE特征生成 ===")
    X_train_df = pd.DataFrame(X_train.toarray() if hasattr(X_train, "toarray") else X_train)
    X_test_df = pd.DataFrame(X_test.toarray() if hasattr(X_test, "toarray") else X_test)
    X_train_df.columns = [str(c) for c in X_train_df.columns]
    X_test_df.columns = [str(c) for c in X_test_df.columns]
    openfe = OpenFE()
    openfe_train, openfe_test = safe_openfe_fit_transform(openfe, X_train_df, y, X_test_df)

    if openfe_train is not None:
        X_train = np.hstack([X_train, openfe_train])
        X_test = np.hstack([X_test, openfe_test])
        print(f"成功生成 {openfe_train.shape[1]} 个OpenFE特征")
    else:
        print("使用原始特征继续流程")
    selector = VarianceThreshold()
    X_train_selected = selector.fit_transform(X_train)
    X_test_selected = selector.transform(X_test)
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_selected, y, test_size=0.2, random_state=42
    )

    models = {
        "LinearRegression": LinearRegression(),
        "XGBoost": XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.05),
        "ExtraTrees": ExtraTreesRegressor(n_estimators=200, random_state=42)
    }

    print("\n=== 模型评估 ===")
    results = {}
    for name, model in models.items():
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_val)
        mae = mean_absolute_error(y_val, y_pred)
        results[name] = mae
        print(f"{name:15} | MAE: {mae:.4f}")

    best_model_name = min(results, key=results.get)
    best_model = models[best_model_name]
    print(f"\n最佳模型: {best_model_name} (MAE: {results[best_model_name]:.4f})")

    final_preds = best_model.predict(X_test_selected)
    final_preds = np.clip(np.round(final_preds), 1, None).astype(int)

    pd.DataFrame({"team_placement": final_preds}).to_csv("submission.csv", index=False)
    print("已保存submission.csv")

if __name__ == '__main__':
    main()
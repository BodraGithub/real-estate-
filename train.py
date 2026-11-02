"""
Real Estate Investment Advisor - training pipeline.

Steps: clean -> engineer features -> create targets -> train classifiers & regressors
       -> track with MLflow -> save models + metadata for the Streamlit app.

Run:   python train.py [path/to/india_housing_prices.csv]
"""
import json
import os
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, mean_absolute_error, mean_squared_error,
                             precision_score, r2_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")
try:
    import xgboost as xgb
except ImportError:
    xgb = None
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
try:
    import mlflow
    import mlflow.sklearn
except ImportError:
    mlflow = None

CURRENT_YEAR, RS, YEARS = 2025, 42, 5
# Business-rule assumptions that define the two targets (change them and re-run)
A = dict(base_growth=5.0, w_infra=3.0, w_under_construction=1.0, w_security=0.5, w_parking=0.5, w_age=-0.04)
AMENITIES = ["Gym", "Pool", "Clubhouse", "Playground", "Garden"]
CAT = ["City", "Property_Type", "Furnished_Status", "Public_Transport_Accessibility", "Facing", "Owner_Type",
       "Availability_Status"]
NUM = ["BHK", "Size_in_SqFt", "Price_in_Lakhs", "Price_per_SqFt_calc", "Floor_No", "Total_Floors", "Floor_Ratio",
       "Age_of_Property", "Nearby_Schools", "Nearby_Hospitals", "Parking_Space", "Security", "Amenities_Count",
       "Size_per_BHK"]
FEATURES = CAT + NUM
OUT, MODELS = "outputs", "models"


def prepare(raw: pd.DataFrame):
    """Clean the data, add features and the two engineered targets. Returns (df, stats)."""
    df = raw.drop_duplicates(subset=[c for c in raw.columns if c != "ID"]).reset_index(drop=True)
    for c in df.select_dtypes(exclude=np.number).columns:                      # imputation + whitespace
        df[c] = df[c].fillna(df[c].mode()[0]).astype(str).str.strip()
    nums = df.select_dtypes(include=np.number).columns.drop("ID", errors="ignore")
    df[nums] = df[nums].fillna(df[nums].median())

    df["Floor_Data_Inconsistent"] = (df["Floor_No"] > df["Total_Floors"]).astype(int)   # impossible floors
    df["Floor_No"] = np.minimum(df["Floor_No"], df["Total_Floors"])
    df["Age_of_Property"] = CURRENT_YEAR - df["Year_Built"]
    df["Price_per_SqFt_calc"] = df["Price_in_Lakhs"] * 1e5 / df["Size_in_SqFt"]          # Rs / sq ft
    q1, q3 = df["Price_per_SqFt_calc"].quantile([0.25, 0.75])
    lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
    df["PPSF_Outlier_Flag"] = ((df["Price_per_SqFt_calc"] < lo) | (df["Price_per_SqFt_calc"] > hi)).astype(int)

    for c in ["Parking_Space", "Security"]:                                              # binary encoding
        df[c] = df[c].map({"Yes": 1, "No": 0}).fillna(0).astype(int)
    df["Amenities_Count"] = df["Amenities"].str.split(",").str.len()
    for a in AMENITIES:
        df[f"Has_{a}"] = df["Amenities"].str.contains(a).astype(int)

    s_min, s_max = df["Nearby_Schools"].min(), df["Nearby_Schools"].max()
    h_min, h_max = df["Nearby_Hospitals"].min(), df["Nearby_Hospitals"].max()
    df["School_Density_Score"] = (df["Nearby_Schools"] - s_min) / max(s_max - s_min, 1)
    df["Hospital_Density_Score"] = (df["Nearby_Hospitals"] - h_min) / max(h_max - h_min, 1)
    df["Transport_Score"] = df["Public_Transport_Accessibility"].map({"Low": 1 / 3, "Medium": 2 / 3, "High": 1.0}).fillna(0.5)
    df["Infra_Score"] = 0.35 * df["School_Density_Score"] + 0.35 * df["Hospital_Density_Score"] + 0.30 * df["Transport_Score"]
    df["Floor_Ratio"] = df["Floor_No"] / df["Total_Floors"].clip(lower=1)
    df["Size_per_BHK"] = df["Size_in_SqFt"] / df["BHK"].clip(lower=1)
    city_median = df.groupby("City")["Price_per_SqFt_calc"].median()
    df["City_Median_PPSF"] = df["City"].map(city_median)
    df["Price_vs_City_Median"] = df["Price_per_SqFt_calc"] / df["City_Median_PPSF"]

    # ---- engineered targets (the dataset has neither) ----
    df["Est_Annual_Growth_Pct"] = (A["base_growth"] + A["w_infra"] * df["Infra_Score"]
                                   + A["w_under_construction"] * (df["Availability_Status"] == "Under_Construction")
                                   + A["w_security"] * df["Security"] + A["w_parking"] * df["Parking_Space"]
                                   + A["w_age"] * df["Age_of_Property"])
    df["Future_Price_5Y"] = df["Price_in_Lakhs"] * (1 + df["Est_Annual_Growth_Pct"] / 100) ** YEARS
    df["Return_5Y_Pct"] = (df["Future_Price_5Y"] / df["Price_in_Lakhs"] - 1) * 100
    threshold = float(df["Est_Annual_Growth_Pct"].median())
    df["Good_Investment"] = ((df["Price_vs_City_Median"] <= 1.0) & (df["Est_Annual_Growth_Pct"] >= threshold)).astype(int)

    stats = dict(
        rows_raw=len(raw), rows_clean=len(df), cols_raw=raw.shape[1], cols_clean=df.shape[1],
        missing_raw=int(raw.isna().sum().sum()), duplicates=int(len(raw) - len(df)),
        floor_inconsistent=int(df["Floor_Data_Inconsistent"].sum()), ppsf_outliers=int(df["PPSF_Outlier_Flag"].sum()),
        growth_threshold=threshold, school_max=int(s_max), hosp_max=int(h_max),
        city_median_ppsf={k: round(float(v), 2) for k, v in city_median.items()},
        city_to_state=df.drop_duplicates("City").set_index("City")["State"].to_dict(),
        categories={c: sorted(df[c].unique().tolist()) for c in CAT},
        size_range=[float(df["Size_in_SqFt"].min()), float(df["Size_in_SqFt"].max())],
        price_range=[float(df["Price_in_Lakhs"].min()), float(df["Price_in_Lakhs"].max())],
        year_range=[int(df["Year_Built"].min()), int(df["Year_Built"].max())],
    )
    return df, stats


def preprocessor():
    return ColumnTransformer([("num", StandardScaler(), NUM),                                  # feature scaling
                              ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT)])  # encoding


def log_mlflow(experiment, name, pipe, metrics, params):
    if mlflow is None:
        return
    try:
        mlflow.set_experiment(experiment)
        with mlflow.start_run(run_name=name):
            mlflow.log_params(params)
            mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
            mlflow.sklearn.log_model(pipe, "model")
    except Exception as e:                                       # tracking must never break training
        print("  (MLflow logging skipped:", e, ")")


def importance(pipe, X, y, scoring):
    sub = X.sample(min(5000, len(X)), random_state=RS)
    r = permutation_importance(pipe, sub, y.loc[sub.index], n_repeats=3, random_state=RS, scoring=scoring, n_jobs=1)
    return pd.Series(r.importances_mean, index=X.columns).sort_values(ascending=False)


def main(path):
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(MODELS, exist_ok=True)
    raw = pd.read_csv(path)
    df, stats = prepare(raw)
    df.to_csv(f"{OUT}/india_housing_cleaned.csv", index=False)
    print(f"Cleaned data: {raw.shape} -> {df.shape} | good investments: {df['Good_Investment'].mean():.1%}")

    X, yc, yr = df[FEATURES], df["Good_Investment"], df["Future_Price_5Y"]
    Xtr, Xte, yc_tr, yc_te, yr_tr, yr_te = train_test_split(X, yc, yr, test_size=0.2, random_state=RS, stratify=yc)

    if mlflow is not None:
        mlflow.set_tracking_uri("sqlite:///mlflow.db")           # view: mlflow ui --backend-store-uri sqlite:///mlflow.db

    # ---------------- classification ----------------
    clf = {"Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
           "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=12, min_samples_leaf=5, n_jobs=-1,
                                                   class_weight="balanced_subsample", random_state=RS)}
    clf["XGBoost" if xgb else "Hist Gradient Boosting"] = (
        xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                          n_jobs=-1, random_state=RS, eval_metric="logloss") if xgb
        else HistGradientBoostingClassifier(max_iter=300, random_state=RS))
    rows, pipes = [], {}
    for name, est in clf.items():
        t0 = time.time()
        pipe = Pipeline([("prep", preprocessor()), ("model", est)]).fit(Xtr, yc_tr)
        proba = pipe.predict_proba(Xte)[:, 1]
        pred = (proba >= 0.5).astype(int)
        m = dict(accuracy=accuracy_score(yc_te, pred), precision=precision_score(yc_te, pred),
                 recall=recall_score(yc_te, pred), f1=f1_score(yc_te, pred), roc_auc=roc_auc_score(yc_te, proba))
        rows.append({"model": name, **m, "train_sec": time.time() - t0})
        pipes[name] = pipe
        log_mlflow("Real_Estate_Classification", name, pipe, m, {"model": name, "task": "good_investment"})
        print(f"[clf] {name:<24} acc={m['accuracy']:.4f} f1={m['f1']:.4f} auc={m['roc_auc']:.4f}")
    clf_df = pd.DataFrame(rows).set_index("model").sort_values("f1", ascending=False)
    best_c = clf_df.index[0]
    cm = confusion_matrix(yc_te, pipes[best_c].predict(Xte)).tolist()
    baseline_acc = float(max(yc_te.mean(), 1 - yc_te.mean()))

    # ---------------- regression ----------------
    reg = {"Linear Regression": LinearRegression(),
           "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=12, min_samples_leaf=5, max_features=0.5,
                                                  n_jobs=-1, random_state=RS)}
    reg["XGBoost" if xgb else "Hist Gradient Boosting"] = (
        xgb.XGBRegressor(n_estimators=400, max_depth=6, learning_rate=0.1, subsample=0.8, colsample_bytree=0.8,
                         n_jobs=-1, random_state=RS) if xgb
        else HistGradientBoostingRegressor(max_iter=400, random_state=RS))
    base_pred = Xte["Price_in_Lakhs"] * (1 + df.loc[Xtr.index, "Est_Annual_Growth_Pct"].mean() / 100) ** YEARS
    rows = [{"model": "Baseline (flat growth)", "rmse": np.sqrt(mean_squared_error(yr_te, base_pred)),
             "mae": mean_absolute_error(yr_te, base_pred), "r2": r2_score(yr_te, base_pred), "train_sec": 0.0}]
    rpipes = {}
    for name, est in reg.items():
        t0 = time.time()
        pipe = Pipeline([("prep", preprocessor()), ("model", est)]).fit(Xtr, yr_tr)
        pred = pipe.predict(Xte)
        m = dict(rmse=np.sqrt(mean_squared_error(yr_te, pred)), mae=mean_absolute_error(yr_te, pred), r2=r2_score(yr_te, pred))
        rows.append({"model": name, **m, "train_sec": time.time() - t0})
        rpipes[name] = pipe
        log_mlflow("Real_Estate_Regression", name, pipe, m, {"model": name, "task": "future_price_5y"})
        print(f"[reg] {name:<24} rmse={m['rmse']:.2f} mae={m['mae']:.2f} r2={m['r2']:.4f}")
    reg_df = pd.DataFrame(rows).set_index("model")
    best_r = reg_df.drop(index="Baseline (flat growth)")["rmse"].idxmin()

    # ---------------- save everything the app needs ----------------
    joblib.dump(pipes[best_c], f"{MODELS}/best_classifier.joblib")
    joblib.dump(rpipes[best_r], f"{MODELS}/best_regressor.joblib")
    clf_df.to_csv(f"{OUT}/classification_results.csv")
    reg_df.to_csv(f"{OUT}/regression_results.csv")
    meta = dict(stats, features=FEATURES, cat_features=CAT, num_features=NUM, assumptions={**A, "years": YEARS},
                best_classifier=best_c, best_regressor=best_r, baseline_accuracy=baseline_acc, confusion_matrix=cm,
                classifier_metrics=clf_df.loc[best_c].to_dict(), regressor_metrics=reg_df.loc[best_r].to_dict(),
                classifier_importance=importance(pipes[best_c], Xte, yc_te, "f1").head(12).to_dict(),
                regressor_importance=importance(rpipes[best_r], Xte, yr_te, "r2").head(12).to_dict())
    with open(f"{MODELS}/app_metadata.json", "w") as f:
        json.dump(meta, f, indent=2, default=float)
    print(f"\nBest classifier: {best_c} | best regressor: {best_r}\nSaved to ./{MODELS} and ./{OUT}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "india_housing_prices.csv")

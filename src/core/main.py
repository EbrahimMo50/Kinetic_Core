from __future__ import annotations

import numpy as np
import pandas as pd
from pandas import DataFrame
 
from imblearn.over_sampling import SMOTE
from sklearn.cluster import DBSCAN, KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import RidgeCV
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import FunctionTransformer, Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBRegressor
 
 
# ---------------------------------------------------------------------------
# Shared constants & helpers
# ---------------------------------------------------------------------------
 
RANDOM_STATE = 42
TEST_SIZE = 0.2
IMBALANCE_RATIO = 0.3  # apply SMOTE if minority/majority falls below this
 
 
def _build_preprocessor(X: DataFrame) -> ColumnTransformer:
    """Median-impute + scale numerics, mode-impute + one-hot encode categoricals."""
    numeric_cols = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_cols = X.select_dtypes(
        include=["object", "string", "category", "bool"]
    ).columns.tolist()
 
    numeric_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ])
 
    categorical_pipeline = Pipeline([
        ("to_object", FunctionTransformer(
            lambda X_: X_.astype(object), feature_names_out="one-to-one"
        )),
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
 
    return ColumnTransformer(
        [
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
    )
 
 
def _prepare_supervised_data(
    df: DataFrame,
    target_column: str,
    *,
    stratify: bool,
    balance: bool,
):
    """Split, preprocess, and (optionally) SMOTE-resample a supervised dataset."""
    X = df.drop(columns=[target_column])
    y = df[target_column]
 
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y if stratify else None,
    )
 
    preprocessor = _build_preprocessor(X)
    X_train_pp = preprocessor.fit_transform(X_train)
    X_test_pp = preprocessor.transform(X_test)
 
    if balance:
        counts = y_train.value_counts()
        if len(counts) > 1 and (counts.min() / counts.max()) < IMBALANCE_RATIO:
            smote = SMOTE(random_state=RANDOM_STATE)
            X_train_pp, y_train = smote.fit_resample(X_train_pp, y_train)
 
    return X_train_pp, X_test_pp, y_train, y_test, preprocessor
 
 
def _flag_best(results: list[dict], metric: str, higher_is_better: bool = True) -> list[dict]:
    """Set ``best_model=True`` on the entry with the best ``report[metric]``."""
    scores = []
    for r in results:
        s = r["report"].get(metric)
        if s is None or (isinstance(s, float) and np.isnan(s)):
            s = float("-inf") if higher_is_better else float("inf")
        scores.append(s)
 
    best_idx = int(np.argmax(scores) if higher_is_better else np.argmin(scores))
    for i, r in enumerate(results):
        r["best_model"] = (i == best_idx)
    return results
 
 
# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------
 
def _regression_report(model, X_test, y_test) -> dict:
    preds = model.predict(X_test)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_test, preds))),
        "mse": float(mean_squared_error(y_test, preds)),
        "mae":  float(mean_absolute_error(y_test, preds)),
        "r2":   float(r2_score(y_test, preds)),
    }
 
 
def get_regression_models(df: DataFrame, target_column: str) -> list[dict]:
    # Regression target is continuous, so no stratification and no SMOTE.
    X_train, X_test, y_train, y_test, _ = _prepare_supervised_data(
        df, target_column, stratify=False, balance=False
    )
 
    # Carve a validation set out of the training data for XGBoost early stopping.
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
 
    xgb_model = XGBRegressor(
        n_estimators=2000,        # ceiling — early stopping will trim this
        learning_rate=0.05,
        early_stopping_rounds=50,
        eval_metric="rmse",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
 
    ridge_model = RidgeCV(alphas=np.logspace(-3, 3, 50), scoring="r2")
    ridge_model.fit(X_train, y_train)
 
    results = [
        {
            "model_type": XGBRegressor,
            "model": xgb_model,
            "report": _regression_report(xgb_model, X_test, y_test),
        },
        {
            "model_type": RidgeCV,
            "model": ridge_model,
            "report": _regression_report(ridge_model, X_test, y_test),
        },
    ]
    return _flag_best(results, metric="r2", higher_is_better=True)
 
 
# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
 
def _classification_report(model, X_test, y_test) -> dict:
    preds = model.predict(X_test)
    return {
        "accuracy":  float(accuracy_score(y_test, preds)),
        "f1":        float(f1_score(y_test, preds, average="weighted")),
        "precision": float(precision_score(y_test, preds, average="weighted", zero_division=0)),
        "recall":    float(recall_score(y_test, preds, average="weighted", zero_division=0)),
    }
 
 
def get_classification_models(df: DataFrame, target_column: str) -> list[dict]:
    X_train, X_test, y_train, y_test, _ = _prepare_supervised_data(
        df, target_column, stratify=True, balance=True
    )
 
    rf_param_grid = {
        "n_estimators":      [200],
        "max_depth":         [None, 10, 20],
        "min_samples_split": [2, 10],
        "min_samples_leaf":  [1, 4],
        "max_features":      ["sqrt"],
    }
    rf_search = GridSearchCV(
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        param_grid=rf_param_grid,
        cv=5,
        scoring="f1_weighted",
        n_jobs=-1,
        verbose=1,
    )
    rf_search.fit(X_train, y_train)
    rf_model = rf_search.best_estimator_
 
    svm_model = SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE)
    svm_model.fit(X_train, y_train)
 
    results = [
        {
            "model_type": RandomForestClassifier,
            "model": rf_model,
            "report": _classification_report(rf_model, X_test, y_test),
        },
        {
            "model_type": SVC,
            "model": svm_model,
            "report": _classification_report(svm_model, X_test, y_test),
        },
    ]
    return _flag_best(results, metric="f1", higher_is_better=True)
 
 
# ---------------------------------------------------------------------------
# Clustering
# ---------------------------------------------------------------------------
 
def _safe_silhouette(X, labels) -> float:
    """Silhouette score that ignores DBSCAN noise points and degenerate cases."""
    mask = labels != -1
    unique_clusters = set(labels[mask]) if mask.any() else set()
    if len(unique_clusters) < 2 or mask.sum() < 2:
        return float("nan")
    return float(silhouette_score(X[mask], labels[mask]))
 
 
def get_clustering_models(df: DataFrame) -> list[dict]:
    # Unsupervised: no target, no train/test split.
    preprocessor = _build_preprocessor(df)
    X_pp = preprocessor.fit_transform(df)
 
    # --- DBSCAN -------------------------------------------------------------
    dbscan = DBSCAN(eps=0.5, min_samples=5)
    db_labels = dbscan.fit_predict(X_pp)
    db_report = {
        "silhouette": _safe_silhouette(X_pp, db_labels),
        "n_clusters": int(len(set(db_labels) - {-1})),
        "n_noise":    int((db_labels == -1).sum()),
    }
 
    # --- KMeans (pick k by silhouette) --------------------------------------
    k_range = list(range(2, 11))
    kmeans_candidates = []
    for k in k_range:
        km = KMeans(n_clusters=k, n_init=10, random_state=RANDOM_STATE)
        labels = km.fit_predict(X_pp)
        kmeans_candidates.append((km, float(silhouette_score(X_pp, labels))))
 
    best_km, best_km_score = max(kmeans_candidates, key=lambda pair: pair[1])
    km_report = {
        "silhouette": best_km_score,
        "n_clusters": int(best_km.n_clusters),
        "n_noise":    0,
    }
 
    results = [
        {"model_type": DBSCAN, "model": dbscan,  "report": db_report},
        {"model_type": KMeans, "model": best_km, "report": km_report},
    ]
    return _flag_best(results, metric="silhouette", higher_is_better=True)
 
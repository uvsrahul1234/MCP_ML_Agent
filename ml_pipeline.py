# ml_pipeline.py
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

import shap

ARTIFACT_DIR = "artifacts"


def _ensure_artifacts_dir() -> None:
    os.makedirs(ARTIFACT_DIR, exist_ok=True)


@dataclass
class DatasetBundle:
    feature_names: list[str]
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


@dataclass
class EvalResult:
    model_name: str
    accuracy: float
    precision: float
    recall: float
    confusion_matrix: list[list[int]]
    classification_report: str


def load_dataset(test_size: float = 0.2, random_state: int = 42) -> DatasetBundle:
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")

    X_train, X_test, y_train, y_test = train_test_split(
        df,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    return DatasetBundle(
        feature_names=list(data.feature_names),
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )


def train_logistic_regression(ds: DatasetBundle, max_iter: int = 10000) -> LogisticRegression:
    model = LogisticRegression(max_iter=max_iter)
    model.fit(ds.X_train, ds.y_train)
    return model


def train_knn(ds: DatasetBundle, n_neighbors: int = 5) -> KNeighborsClassifier:
    model = KNeighborsClassifier(n_neighbors=n_neighbors)
    model.fit(ds.X_train, ds.y_train)
    return model


def save_model(model: Any, model_name: str) -> str:
    _ensure_artifacts_dir()
    path = os.path.join(ARTIFACT_DIR, f"{model_name}.joblib")
    joblib.dump(model, path)
    return path


def load_model(model_name: str) -> Any:
    path = os.path.join(ARTIFACT_DIR, f"{model_name}.joblib")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model artifact not found: {path}")
    return joblib.load(path)


def predict_with_model(model: Any, input_data: Dict[str, Any]) -> Dict[str, Any]:
    X = pd.DataFrame([input_data])

    pred = int(model.predict(X)[0])

    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        confidence = float(np.max(proba))
    else:
        confidence = None

    return {
        "prediction": pred,
        "confidence": confidence,
    }


def evaluate_model(model: Any, model_name: str, ds: DatasetBundle) -> EvalResult:
    y_pred = model.predict(ds.X_test)

    acc = float(accuracy_score(ds.y_test, y_pred))
    prec = float(precision_score(ds.y_test, y_pred))
    rec = float(recall_score(ds.y_test, y_pred))

    cm = confusion_matrix(ds.y_test, y_pred)

    report = classification_report(ds.y_test, y_pred)

    return EvalResult(
        model_name=model_name,
        accuracy=acc,
        precision=prec,
        recall=rec,
        confusion_matrix=cm.tolist(),
        classification_report=report,
    )


def select_model(
    eval_lr: EvalResult,
    eval_knn: EvalResult,
    rule: str = "recall_then_precision_then_accuracy",
) -> Dict[str, Any]:
    """
    Model selection tool.
    Default rule is suitable for the breast cancer context:
    prioritize recall, then precision, then accuracy.
    """
    candidates = [eval_lr, eval_knn]

    if rule == "accuracy":
        best = max(candidates, key=lambda r: r.accuracy)
        rationale = "Selected model with highest accuracy."
    elif rule == "precision":
        best = max(candidates, key=lambda r: r.precision)
        rationale = "Selected model with highest precision."
    elif rule == "recall":
        best = max(candidates, key=lambda r: r.recall)
        rationale = "Selected model with highest recall."
    else:
        best = max(candidates, key=lambda r: (r.recall, r.precision, r.accuracy))
        rationale = "Selected model with highest recall; ties broken by precision, then accuracy."

    selection = {
        "selected_model": best.model_name,
        "rule": rule,
        "rationale": rationale,
        "metrics": {
            best.model_name: {
                "accuracy": best.accuracy,
                "precision": best.precision,
                "recall": best.recall,
            }
        },
    }

    _ensure_artifacts_dir()
    with open(os.path.join(ARTIFACT_DIR, "selection.json"), "w", encoding="utf-8") as f:
        json.dump(selection, f, indent=2)

    return selection


def shap_explain(
    model: Any,
    model_name: str,
    ds: DatasetBundle,
    local_index: int = 0,
    top_k: int = 8,
) -> Dict[str, Any]:
    """
    SHAP tool: returns global top features and one local explanation.
    We return data, not images, so it stays tool-friendly.
    """
    explainer = shap.Explainer(model, ds.X_train)
    sv = explainer(ds.X_test)

    # Global: mean absolute shap by feature
    vals = np.abs(sv.values)
    mean_abs = vals.mean(axis=0)
    global_rank = np.argsort(mean_abs)[::-1]

    global_top = []
    for idx in global_rank[:top_k]:
        global_top.append({
            "feature": ds.X_test.columns[idx],
            "mean_abs_shap": float(mean_abs[idx]),
        })

    # Local: single row contributions
    local_index = int(local_index)
    local_index = max(0, min(local_index, ds.X_test.shape[0] - 1))

    local_features = []
    local_vals = sv.values[local_index]
    row = ds.X_test.iloc[local_index]

    # top contributors by absolute value
    local_rank = np.argsort(np.abs(local_vals))[::-1][:top_k]
    for idx in local_rank:
        local_features.append({
            "feature": ds.X_test.columns[idx],
            "feature_value": float(row.iloc[idx]),
            "shap_value": float(local_vals[idx]),
        })

    explanation = {
        "model_name": model_name,
        "global_top_features": global_top,
        "local_index": local_index,
        "local_top_contributions": local_features,
    }

    _ensure_artifacts_dir()
    with open(os.path.join(ARTIFACT_DIR, f"shap_{model_name}.json"), "w", encoding="utf-8") as f:
        json.dump(explanation, f, indent=2)

    return explanation


def train_and_export_all(
    test_size: float = 0.2,
    random_state: int = 42,
    knn_neighbors: int = 5,
) -> Dict[str, Any]:
    """
    Convenience function for the "train models and export joblib" step.
    """
    ds = load_dataset(test_size=test_size, random_state=random_state)

    lr = train_logistic_regression(ds)
    knn = train_knn(ds, n_neighbors=knn_neighbors)

    lr_path = save_model(lr, "logistic_regression")
    knn_path = save_model(knn, "knn")

    # Save feature schema for validation at tool-call time
    _ensure_artifacts_dir()
    schema_path = os.path.join(ARTIFACT_DIR, "feature_names.json")
    with open(schema_path, "w", encoding="utf-8") as f:
        json.dump(ds.feature_names, f, indent=2)

    return {
        "status": "trained_and_exported",
        "artifacts": {
            "logistic_regression": lr_path,
            "knn": knn_path,
            "feature_schema": schema_path,
        },
        "dataset": {
            "train_rows": int(ds.X_train.shape[0]),
            "test_rows": int(ds.X_test.shape[0]),
        }
    }
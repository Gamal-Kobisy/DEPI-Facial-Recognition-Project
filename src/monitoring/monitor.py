"""
MLOps monitoring utilities for the Facial Recognition System.

Tracks key performance metrics (accuracy, FAR) over time using MLflow
and raises alerts when metric values fall below defined thresholds.
"""

import datetime
import os

import numpy as np

try:
    import mlflow
    _MLFLOW_AVAILABLE = True
except ImportError:
    _MLFLOW_AVAILABLE = False


# Default alert thresholds
DEFAULT_THRESHOLDS = {
    "accuracy": 0.90,   # Minimum acceptable accuracy
    "far": 0.05,        # Maximum acceptable False Acceptance Rate
}


def log_metrics(
    accuracy: float,
    far: float,
    step: int | None = None,
    run_name: str | None = None,
) -> None:
    """
    Log model performance metrics to MLflow.

    Parameters
    ----------
    accuracy : float
        Model accuracy on the evaluation set.
    far : float
        False Acceptance Rate.
    step : int or None
        Optional training step / epoch number for time-series tracking.
    run_name : str or None
        Optional MLflow run name.
    """
    if not _MLFLOW_AVAILABLE:
        print(
            f"[monitor] accuracy={accuracy:.4f}, FAR={far:.4f} "
            "(mlflow not installed – metrics not persisted)"
        )
        return

    with mlflow.start_run(run_name=run_name or f"eval_{datetime.datetime.now(datetime.timezone.utc).isoformat()}"):
        mlflow.log_metric("accuracy", accuracy, step=step)
        mlflow.log_metric("far", far, step=step)


def check_thresholds(
    accuracy: float,
    far: float,
    thresholds: dict | None = None,
) -> list:
    """
    Compare metric values against acceptable thresholds.

    Parameters
    ----------
    accuracy : float
        Current model accuracy.
    far : float
        Current False Acceptance Rate.
    thresholds : dict or None
        Override the default thresholds.  Keys: ``"accuracy"``, ``"far"``.

    Returns
    -------
    list of str
        Alert messages for each violated threshold.  Empty list means all
        metrics are within acceptable bounds.
    """
    thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    alerts = []

    if accuracy < thresholds["accuracy"]:
        alerts.append(
            f"ALERT: accuracy {accuracy:.4f} is below threshold {thresholds['accuracy']:.4f}. "
            "Model retraining recommended."
        )

    if far > thresholds["far"]:
        alerts.append(
            f"ALERT: FAR {far:.4f} exceeds threshold {thresholds['far']:.4f}. "
            "Security risk – model retraining required."
        )

    return alerts


def run_monitoring_cycle(
    accuracy: float,
    far: float,
    step: int | None = None,
    thresholds: dict | None = None,
) -> list:
    """
    Execute a full monitoring cycle: log metrics and check thresholds.

    Parameters
    ----------
    accuracy : float
        Current model accuracy.
    far : float
        Current False Acceptance Rate.
    step : int or None
        Optional step/epoch for MLflow logging.
    thresholds : dict or None
        Custom threshold overrides.

    Returns
    -------
    list of str
        Alert messages (empty if all metrics are healthy).
    """
    log_metrics(accuracy=accuracy, far=far, step=step)
    alerts = check_thresholds(accuracy=accuracy, far=far, thresholds=thresholds)

    for alert in alerts:
        print(alert)

    return alerts

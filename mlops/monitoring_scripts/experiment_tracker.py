"""
Milestone 4 - MLOps & Monitoring
File: mlops/monitoring_scripts/experiment_tracker.py

Integrated with the project's existing thresholds:
  - BLACKLIST_THRESHOLD = 0.55  (from stream_scanner.py)
  - VISITOR_THRESHOLD   = 0.55  (from stream_scanner.py)
  - Embedding dim = 128, distance = cosine  (from core_logic.py)
"""

import os
import json
import logging
import numpy as np
from datetime import datetime
from pathlib import Path

import mlflow
import mlflow.keras
from mlflow.tracking import MlflowClient


THIS_FILE   = os.path.abspath(__file__)
SCRIPTS_DIR = os.path.dirname(THIS_FILE)           # monitoring_scripts/
MLOPS_DIR   = os.path.dirname(SCRIPTS_DIR)          # mlops/
PROJECT_ROOT = os.path.dirname(MLOPS_DIR)           # project root
MLFLOW_DB   = os.path.join(PROJECT_ROOT, "mlflow", "mlflow.db")
MLFLOW_ART  = os.path.join(PROJECT_ROOT, "mlflow", "artifacts")


os.makedirs(os.path.join(PROJECT_ROOT, "logs"), exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(
            os.path.join(PROJECT_ROOT, "logs", "experiment_tracker.log"),
            mode="a", encoding="utf-8"
        ),
    ],
)
logger = logging.getLogger("ExperimentTracker")


MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", f"sqlite:///{MLFLOW_DB}")
MLFLOW_ARTIFACT_URI = os.getenv("MLFLOW_ARTIFACT_URI", MLFLOW_ART)
EXPERIMENT_NAME     = "Facial_Recognition_Production"
MODEL_NAME          = "FaceNet_DeepFace_Security"


BLACKLIST_THRESHOLD = 0.55
VISITOR_THRESHOLD   = 0.55
COOLDOWN_SECONDS    = 60


FAR_THRESHOLD  = 0.02
FRR_THRESHOLD  = 0.05
ACC_THRESHOLD  = 0.90
F1_THRESHOLD   = 0.88


class ExperimentTracker:
    """
    MLflow wrapper for Intelligent Shop Security.
    Always reads/writes to the same DB regardless of working directory.
    """

    def __init__(self):
        os.makedirs(os.path.join(PROJECT_ROOT, "mlflow"), exist_ok=True)
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        self.client     = MlflowClient()
        self.experiment = self._get_or_create_experiment()
        self.active_run = None
        logger.info(f"ExperimentTracker ready - experiment: '{EXPERIMENT_NAME}'")
        logger.info(f"DB path: {MLFLOW_DB}")

    def _get_or_create_experiment(self):
        exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        if exp is None:
            eid = mlflow.create_experiment(
                EXPERIMENT_NAME,
                artifact_location=MLFLOW_ARTIFACT_URI,
                tags={
                    "project":   "DEPI-Facial-Recognition",
                    "model":     "FaceNet",
                    "framework": "DeepFace",
                },
            )
            exp = mlflow.get_experiment(eid)
            logger.info(f"Created experiment id={eid}")
        return exp


    def start_run(self, run_name: str = None, tags: dict = None):
        run_name = run_name or f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        base_tags = {
            "milestone":           "4",
            "blacklist_threshold": str(BLACKLIST_THRESHOLD),
            "visitor_threshold":   str(VISITOR_THRESHOLD),
            "embedding_dim":       "128",
            "distance_metric":     "cosine",
        }
        if tags:
            base_tags.update(tags)
        self.active_run = mlflow.start_run(
            experiment_id=self.experiment.experiment_id,
            run_name=run_name,
            tags=base_tags,
        )
        logger.info(f"Run started - '{run_name}'  id={self.active_run.info.run_id}")
        return self

    def end_run(self, status: str = "FINISHED"):
        if self.active_run:
            mlflow.end_run(status=status)
            self.active_run = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, *_):
        self.end_run("FAILED" if exc_type else "FINISHED")


    def log_params(self, params: dict):
        base = {
            "model_architecture":  "FaceNet",
            "framework":           "DeepFace",
            "embedding_dim":       128,
            "distance_metric":     "cosine",
            "blacklist_threshold": BLACKLIST_THRESHOLD,
            "visitor_threshold":   VISITOR_THRESHOLD,
            "target_size":         "160x160",
        }
        base.update(params)
        mlflow.log_params(base)
        logger.info(f"Logged {len(base)} param(s)")

    def log_metrics(self, metrics: dict, step: int = None) -> bool:
        mlflow.log_metrics(metrics, step=step)
        violations = []
        for name, val, op, thresh in [
            ("far",      metrics.get("far",      0), ">", FAR_THRESHOLD),
            ("accuracy", metrics.get("accuracy", 1), "<", ACC_THRESHOLD),
            ("f1_score", metrics.get("f1_score", 1), "<", F1_THRESHOLD),
        ]:
            if (val > thresh if op == ">" else val < thresh):
                violations.append(f"{name}={val:.4f} (threshold {op} {thresh})")

        if violations:
            mlflow.set_tag("threshold_violations", " | ".join(violations))
            logger.warning(f"[WARN] Violations: {violations}")
        else:
            mlflow.set_tag("threshold_status", "PASS")
            logger.info("[PASS] All thresholds passed")
        return len(violations) == 0

    def log_model(self, model, artifact_path: str = "model"):
        mlflow.keras.log_model(model, artifact_path)

    def log_artifact(self, local_path: str):
        mlflow.log_artifact(local_path)

    def log_dict(self, data: dict, filename: str):
        mlflow.log_dict(data, filename)


    def register_model(self, run_id: str = None, artifact_path: str = "model"):
        rid = run_id or self.active_run.info.run_id
        result = mlflow.register_model(f"runs:/{rid}/{artifact_path}", MODEL_NAME)
        logger.info(f"Registered -> version {result.version}")
        return result

    def promote_to_production(self, version: int):
        self.client.transition_model_version_stage(
            name=MODEL_NAME, version=str(version),
            stage="Production", archive_existing_versions=True,
        )
        logger.info(f"Version {version} promoted to Production")

    def register_model_if_best(self, current_metrics: dict) -> bool:
        current_far = current_metrics.get("far", 1.0)
        try:
            prod = self.client.get_latest_versions(MODEL_NAME, stages=["Production"])
            if prod:
                prod_far = float(
                    self.client.get_run(prod[0].run_id).data.metrics.get("far", 1.0)
                )
                logger.info(f"Prod FAR={prod_far:.4f}  |  New FAR={current_far:.4f}")
                if current_far >= prod_far:
                    logger.info("Not better than production - skipping")
                    return False
        except Exception:
            logger.info("No production model yet - registering first")

        result = self.register_model()
        self.promote_to_production(result.version)
        return True

    def get_recent_runs(self, n: int = 10) -> list:
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment.experiment_id],
            order_by=["start_time DESC"],
            max_results=n,
        )
        return runs.to_dict("records") if not runs.empty else []

    def generate_experiment_summary(self) -> dict:
        runs = self.get_recent_runs(50)
        if not runs:
            return {"message": "No runs found"}
        summary = {"total_runs": len(runs), "metrics": {}}
        for key in ["accuracy", "far", "f1_score", "precision", "recall"]:
            col  = f"metrics.{key}"
            vals = [r[col] for r in runs if col in r and r[col] is not None]
            if vals:
                summary["metrics"][key] = {
                    "best":   round(min(vals) if key == "far" else max(vals), 4),
                    "latest": round(vals[0], 4),
                    "mean":   round(float(np.mean(vals)), 4),
                }
        return summary


if __name__ == "__main__":
    tracker = ExperimentTracker()
    demo_metrics = {
        "accuracy": 0.9450, "precision": 0.9380, "recall": 0.9510,
        "f1_score": 0.9444, "far": 0.0120, "frr": 0.0310, "auc": 0.9820,
    }
    demo_params = {
        "optimizer": "adam", "learning_rate": 1e-4,
        "batch_size": 32, "epochs": 10,
        "augmentation": True, "dataset": "LFW+VGGFace2",
    }
    with tracker.start_run("demo_m4_run", tags={"triggered": "manual"}):
        tracker.log_params(demo_params)
        tracker.log_metrics(demo_metrics)
        tracker.log_dict(demo_metrics, "eval_metrics.json")

    print(json.dumps(tracker.generate_experiment_summary(), indent=2))

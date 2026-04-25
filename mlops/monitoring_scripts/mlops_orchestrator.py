"""
Milestone 4 – MLOps & Monitoring
File: mlops/monitoring_scripts/mlops_orchestrator.py
"""

import os
import sys
import json
import logging
import threading
from datetime import datetime
from pathlib import Path

from flask import Flask, jsonify, request
from flask_cors import CORS


THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from experiment_tracker    import ExperimentTracker
from retraining_pipeline   import RetrainingPipeline
from monitor               import PerformanceMonitor


os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/orchestrator.log", mode="a"),
    ],
)
logger = logging.getLogger("MLOpsOrchestrator")

MLOPS_PORT = 5002


class MLOpsOrchestrator:
    def __init__(self, ai_engine_url: str = "http://localhost:5001"):
        logger.info("Initialising MLOps Orchestrator …")

        self.tracker  = ExperimentTracker()
        self.pipeline = RetrainingPipeline(tracker=self.tracker)

        def _retrain_cb(metrics):
            logger.info("Auto-retrain triggered by monitor")
            self.pipeline.run(metrics)

        self.monitor = PerformanceMonitor(
            retrain_callback=_retrain_cb,
            ai_engine_url=ai_engine_url,
        )
        self._lock = threading.Lock()
        logger.info("✅ MLOps Orchestrator ready")

    def start_monitoring(self):
        self.monitor.start()

    def stop(self):
        self.monitor.stop()

    def get_status(self) -> dict:
        check = self.monitor.run_once()
        return {
            "system":         "Intelligent Shop Security – MLOps",
            "milestone":      "4",
            "timestamp":      datetime.now().isoformat(),
            "monitor_status": check["status"],
            "metrics":        check["metrics"],
            "violations":     check["violations"],
            "rolling":        check["rolling"],
        }

    def trigger_retrain(self, force: bool = False) -> dict:
        with self._lock:
            check = self.monitor.run_once()
            return self.pipeline.run(check["metrics"], force=force)

    def list_experiments(self, n: int = 10) -> dict:
        return {
            "runs":    self.tracker.get_recent_runs(n),
            "summary": self.tracker.generate_experiment_summary(),
        }

    def generate_report(self) -> str:
        return self.monitor.generate_report()



def create_app(orch: MLOpsOrchestrator) -> Flask:
    app = Flask(__name__)
    CORS(app, resources={r"/api/*": {"origins": "*"}})

    @app.route("/api/mlops/health")
    def health():
        return jsonify({"status": "ok", "port": MLOPS_PORT,
                        "service": "MLOps Orchestrator"})

    @app.route("/api/mlops/status")
    def status():
        try:
            return jsonify(orch.get_status())
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/mlops/metrics")
    def metrics():
        return jsonify(orch.monitor.dashboard_data())

    @app.route("/api/mlops/retrain", methods=["POST"])
    def retrain():
        try:
            body  = request.get_json(silent=True) or {}
            force = body.get("force", False)
            return jsonify(orch.trigger_retrain(force=force))
        except Exception as e:
            return jsonify({"error": str(e)}), 500

    @app.route("/api/mlops/experiments")
    def experiments():
        n = int(request.args.get("n", 10))
        return jsonify(orch.list_experiments(n))

    @app.route("/api/mlops/report")
    def report():
        path = orch.generate_report()
        return jsonify({"report_path": path, "status": "generated"})

    @app.route("/api/mlops/alerts")
    def alerts():
        alerts_path = Path("logs/alerts.jsonl")
        items = []
        if alerts_path.exists():
            with open(alerts_path) as f:
                for line in f.readlines()[-50:]:
                    try:
                        items.append(json.loads(line))
                    except Exception:
                        pass
        return jsonify({"alerts": items[::-1]})

    return app



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",       type=int, default=MLOPS_PORT)
    parser.add_argument("--engine",     default="http://localhost:5001")
    parser.add_argument("--no-monitor", action="store_true")
    args = parser.parse_args()

    orch = MLOpsOrchestrator(ai_engine_url=args.engine)
    if not args.no_monitor:
        orch.start_monitoring()
        logger.info("Background monitor daemon started")

    app = create_app(orch)
    logger.info(f"MLOps API → http://localhost:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=False)

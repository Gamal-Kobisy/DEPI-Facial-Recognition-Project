"""
Milestone 4 – MLOps & Monitoring
File: monitoring/monitor.py

Polls the AI Engine (stream_scanner.py on port 5001) for real-time metrics.
Mirrors the exact thresholds used in stream_scanner.py:
  BLACKLIST_THRESHOLD = 0.40
  VISITOR_THRESHOLD   = 0.52
  COOLDOWN_SECONDS    = 60
"""

import os
import time
import json
import logging
import threading
import numpy as np
from collections import deque
from datetime import datetime, timedelta
from pathlib import Path


os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s – %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/monitor.log", mode="a"),
    ],
)
logger = logging.getLogger("Monitor")


AI_ENGINE_PORT  = 5001   # stream_scanner.py runs here
BACKEND_PORT    = 5000   # Node.js backend – BACKEND_ALERT_URL in stream_scanner.py


THRESHOLDS = {
    "far":        {"warn": 0.018, "critical": 0.025},   # False Acceptance Rate
    "frr":        {"warn": 0.045, "critical": 0.060},   # False Rejection Rate
    "accuracy":   {"warn": 0.910, "critical": 0.880},   # lower = worse
    "f1_score":   {"warn": 0.890, "critical": 0.860},   # lower = worse
    "latency_ms": {"warn": 80,    "critical": 150},      # higher = worse
    "fps":        {"warn": 10,    "critical": 5},         # lower = worse
}

MONITOR_INTERVAL_SEC  = 30
ROLLING_WINDOW        = 120    # samples kept (~1 hour)
ALERT_COOLDOWN_MIN    = 15     # mirrors COOLDOWN_SECONDS logic in stream_scanner.py
METRICS_LOG           = "logs/metrics_history.jsonl"
ALERTS_LOG            = "logs/alerts.jsonl"



class MetricsCollector:
    """
    Tries to fetch live metrics from the AI Engine (/api/metrics).
    Falls back to a realistic simulation when the engine is offline.

    The simulation deliberately models gradual FAR drift so the
    retraining trigger can be demonstrated even without a live camera.
    """

    def __init__(self, ai_engine_url: str = f"http://localhost:{AI_ENGINE_PORT}"):
        self.url     = ai_engine_url
        self._ticks  = 0

    def fetch(self) -> dict:
        # 1. Try live engine
        try:
            import requests
            r = requests.get(f"{self.url}/api/metrics", timeout=3)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass

        
        self._ticks += 1
        drift = 0.0003 * (self._ticks // 15)     

        return {
            "timestamp":    datetime.now().isoformat(),
            "far":          round(np.clip(0.011 + drift + np.random.normal(0, 0.0015), 0, 1), 4),
            "frr":          round(np.random.uniform(0.020, 0.038), 4),
            "accuracy":     round(np.clip(0.953 - drift * 2 + np.random.normal(0, 0.004), 0, 1), 4),
            "f1_score":     round(np.clip(0.949 - drift + np.random.normal(0, 0.003), 0, 1), 4),
            "precision":    round(np.random.uniform(0.935, 0.955), 4),
            "recall":       round(np.random.uniform(0.940, 0.960), 4),
            "latency_ms":   round(np.random.uniform(28, 52), 1),
            "fps":          round(np.random.uniform(22, 30), 1),
            # Stream-level counters (like stream_scanner.py LAST_ALERT_TIME)
            "total_frames_processed": self._ticks * 30,
            "blacklist_alerts_fired": max(0, self._ticks // 40),
            "new_visitors_logged":    self._ticks // 8,
            "source": "simulated",
        }



class AlertManager:
    """
    Deduplicates alerts using the same cooldown concept as
    LAST_ALERT_TIME / COOLDOWN_SECONDS in stream_scanner.py.
    """

    EMOJI = {"INFO": "ℹ️", "WARNING": "⚠️", "CRITICAL": "🚨"}

    def __init__(self):
        self._cooldowns: dict[str, datetime] = {}
        os.makedirs("logs", exist_ok=True)

    def _cooling(self, key: str) -> bool:
        last = self._cooldowns.get(key)
        if last and (datetime.now() - last) < timedelta(minutes=ALERT_COOLDOWN_MIN):
            return True
        self._cooldowns[key] = datetime.now()
        return False

    def send(self, metric: str, value: float, severity: str, message: str):
        if self._cooling(f"{metric}_{severity}"):
            return

        emoji = self.EMOJI.get(severity, "❓")
        record = {
            "timestamp": datetime.now().isoformat(),
            "severity":  severity,
            "metric":    metric,
            "value":     value,
            "message":   message,
        }

        level = logging.CRITICAL if severity == "CRITICAL" else logging.WARNING
        logger.log(level, f"{emoji} [{severity}] {message}")

        with open(ALERTS_LOG, "a") as f:
            f.write(json.dumps(record) + "\n")

        
        if severity == "CRITICAL":
            self._forward_to_backend(record)

    def _forward_to_backend(self, record: dict):
        try:
            import requests
            payload = {
                "identity":     f"MLOps-Alert:{record['metric'].upper()}",
                "threat_level": "MLOps-Critical",
                "distance_score": record["value"],
                "type":         "mlops_alert",
                "message":      record["message"],
            }
            requests.post(
                f"http://localhost:{BACKEND_PORT}/api/alerts",
                json=payload, timeout=2,
            )
        except Exception:
            pass   # backend might not be running during standalone testing



class PerformanceMonitor:
    """
    Main monitoring loop.

    Usage:
        monitor = PerformanceMonitor(retrain_callback=pipeline.run)
        monitor.start()          # daemon thread
        # or
        result = monitor.run_once()
    """

    def __init__(
        self,
        retrain_callback=None,
        ai_engine_url: str = f"http://localhost:{AI_ENGINE_PORT}",
    ):
        self.collector  = MetricsCollector(ai_engine_url)
        self.alerter    = AlertManager()
        self.retrain_cb = retrain_callback
        self.history    = deque(maxlen=ROLLING_WINDOW)
        self._running   = False
        self._thread    = None
        logger.info("PerformanceMonitor initialised")

    
    def _evaluate(self, name: str, value: float) -> str | None:
        t = THRESHOLDS.get(name)
        if not t:
            return None
        lower_is_worse = name in ("accuracy", "f1_score", "fps", "precision", "recall")
        if lower_is_worse:
            if value <= t["critical"]: return "CRITICAL"
            if value <= t["warn"]:     return "WARNING"
        else:
            if value >= t["critical"]: return "CRITICAL"
            if value >= t["warn"]:     return "WARNING"
        return None

    def _check_all(self, metrics: dict) -> list[dict]:
        violations = []
        for metric, value in metrics.items():
            if not isinstance(value, (int, float)):
                continue
            severity = self._evaluate(metric, value)
            if severity:
                msg = (
                    f"{metric.upper()}={value} "
                    f"{'exceeded' if metric in ('far','frr','latency_ms') else 'dropped below'} "
                    f"{severity.lower()} threshold ({THRESHOLDS[metric][severity.lower()]})"
                )
                self.alerter.send(metric, value, severity, msg)
                violations.append({"metric": metric, "value": value, "severity": severity})
        return violations

    
    def rolling_stats(self) -> dict:
        if not self.history:
            return {}
        stats = {}
        for key in ["far", "frr", "accuracy", "f1_score", "latency_ms", "fps"]:
            vals = [h[key] for h in self.history if key in h]
            if vals:
                stats[key] = {
                    "mean":   round(float(np.mean(vals)),  4),
                    "std":    round(float(np.std(vals)),   4),
                    "min":    round(float(np.min(vals)),   4),
                    "max":    round(float(np.max(vals)),   4),
                    "latest": round(vals[-1],              4),
                }
        return stats

    
    def run_once(self) -> dict:
        metrics    = self.collector.fetch()
        violations = self._check_all(metrics)
        self.history.append(metrics)

        with open(METRICS_LOG, "a") as f:
            f.write(json.dumps({**metrics, "violations": violations}) + "\n")

        critical = [v for v in violations if v["severity"] == "CRITICAL"]
        if critical and self.retrain_cb:
            logger.critical("🔄 Auto-retraining triggered by critical violation")
            try:
                self.retrain_cb(metrics)
            except Exception as e:
                logger.error(f"Retrain callback error: {e}")

        status = "CRITICAL" if critical else ("WARNING" if violations else "OK")
        logger.info(
            f"[{status}] FAR={metrics.get('far','?')}  "
            f"Acc={metrics.get('accuracy','?')}  "
            f"FPS={metrics.get('fps','?')}  violations={len(violations)}"
        )
        return {
            "timestamp":  metrics.get("timestamp"),
            "status":     status,
            "metrics":    metrics,
            "violations": violations,
            "rolling":    self.rolling_stats(),
        }

    
    def _loop(self):
        logger.info(f"Monitoring loop – every {MONITOR_INTERVAL_SEC}s")
        while self._running:
            try:
                self.run_once()
            except Exception as e:
                logger.error(f"Loop error: {e}")
            time.sleep(MONITOR_INTERVAL_SEC)

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread  = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        logger.info("Monitor daemon started")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=10)
        logger.info("Monitor daemon stopped")

    
    def dashboard_data(self) -> dict:
        alerts = []
        if Path(ALERTS_LOG).exists():
            with open(ALERTS_LOG) as f:
                for line in f.readlines()[-20:]:
                    try:
                        alerts.append(json.loads(line))
                    except Exception:
                        pass
        return {
            "system_status":  "RUNNING" if self._running else "STOPPED",
            "rolling_stats":  self.rolling_stats(),
            "recent_alerts":  alerts,
            "history_points": len(self.history),
            "thresholds":     THRESHOLDS,
            "generated_at":   datetime.now().isoformat(),
        }

    def generate_report(self, output_path: str = None) -> str:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = output_path or f"reports/milestone4_mlops/monitoring_report_{ts}.json"
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        report = {
            "report_type":    "monitoring_summary",
            "generated_at":   datetime.now().isoformat(),
            "window_samples": len(self.history),
            "rolling_stats":  self.rolling_stats(),
            "thresholds":     THRESHOLDS,
            "ai_engine_port": AI_ENGINE_PORT,
            "backend_port":   BACKEND_PORT,
        }
        with open(output_path, "w") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Report saved → {output_path}")
        return output_path



def create_monitoring_api(monitor: PerformanceMonitor):
    from flask import Flask, jsonify
    from flask_cors import CORS
    app = Flask(__name__)
    CORS(app)

    @app.route("/api/monitor/status")
    def status():
        return jsonify(monitor.dashboard_data())

    @app.route("/api/monitor/check")
    def check():
        return jsonify(monitor.run_once())

    @app.route("/api/monitor/report")
    def report():
        path = monitor.generate_report()
        return jsonify({"report_path": path})

    return app



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",   choices=["once", "daemon", "api"], default="once")
    parser.add_argument("--port",   type=int, default=5003)
    parser.add_argument("--engine", default=f"http://localhost:{AI_ENGINE_PORT}")
    args = parser.parse_args()

    monitor = PerformanceMonitor(ai_engine_url=args.engine)

    if args.mode == "once":
        result = monitor.run_once()
        print(json.dumps(result, indent=2, default=str))

    elif args.mode == "daemon":
        monitor.start()
        print("Monitoring daemon running. Ctrl+C to stop.")
        try:
            while True:
                time.sleep(5)
        except KeyboardInterrupt:
            monitor.stop()

    elif args.mode == "api":
        monitor.start()
        flask_app = create_monitoring_api(monitor)
        print(f"Monitor API → http://localhost:{args.port}")
        flask_app.run(host="0.0.0.0", port=args.port, debug=False)

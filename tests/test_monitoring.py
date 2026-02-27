"""
Tests for monitoring utilities.
"""

import pytest
import numpy as np

from src.monitoring.monitor import check_thresholds, DEFAULT_THRESHOLDS


class TestCheckThresholds:
    def test_no_alerts_when_metrics_healthy(self):
        alerts = check_thresholds(accuracy=0.95, far=0.02)
        assert alerts == []

    def test_alert_on_low_accuracy(self):
        alerts = check_thresholds(accuracy=0.80, far=0.02)
        assert len(alerts) == 1
        assert "accuracy" in alerts[0].lower()

    def test_alert_on_high_far(self):
        alerts = check_thresholds(accuracy=0.95, far=0.10)
        assert len(alerts) == 1
        assert "far" in alerts[0].lower()

    def test_two_alerts_when_both_fail(self):
        alerts = check_thresholds(accuracy=0.50, far=0.20)
        assert len(alerts) == 2

    def test_custom_thresholds_override(self):
        # With a very strict accuracy threshold, a good model should still alert
        alerts = check_thresholds(accuracy=0.92, far=0.01, thresholds={"accuracy": 0.99})
        assert any("accuracy" in a.lower() for a in alerts)

    def test_boundary_values_do_not_alert(self):
        acc_threshold = DEFAULT_THRESHOLDS["accuracy"]
        far_threshold = DEFAULT_THRESHOLDS["far"]
        alerts = check_thresholds(accuracy=acc_threshold, far=far_threshold)
        assert alerts == []

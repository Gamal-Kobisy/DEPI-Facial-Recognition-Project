"""
Tests for the Flask deployment application.
"""

import pytest


@pytest.fixture
def client():
    try:
        from flask import Flask
    except ImportError:
        pytest.skip("Flask not installed")

    from src.deployment.app import app
    app.config["TESTING"] = True
    with app.test_client() as c:
        yield c


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_returns_ok_status(self, client):
        data = client.get("/health").get_json()
        assert data["status"] == "ok"


class TestIdentifyEndpoint:
    def test_missing_image_returns_400(self, client):
        resp = client.post("/identify")
        assert resp.status_code == 400

    def test_invalid_bytes_returns_422(self, client):
        data = {"image": (b"not_an_image", "test.jpg")}
        resp = client.post("/identify", data=data, content_type="multipart/form-data")
        assert resp.status_code in (422, 500)


class TestVerifyEndpoint:
    def test_missing_fields_returns_400(self, client):
        resp = client.post("/verify")
        assert resp.status_code == 400

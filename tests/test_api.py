"""
Test cases for the SoftwareTeamFabrik REST API.
"""

import pytest
from fastapi.testclient import TestClient
from factory.adapters.api import app, CommandResponse
import json

def test_health_check():
    """Test the health check endpoint"""
    client = TestClient(app)
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "SoftwareTeamFabrik API is running" in data["message"]

def test_status_endpoint():
    """Test the status endpoint"""
    client = TestClient(app)
    response = client.get("/api/v1/status")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "Status retrieved successfully" in data["message"]
    assert "data" in data

def test_command_response_model():
    """Test the CommandResponse model"""
    response = CommandResponse(
        success=True,
        message="Test message",
        data={"key": "value"},
        error=None
    )
    
    assert response.success is True
    assert response.message == "Test message"
    assert response.data == {"key": "value"}
    assert response.error is None

def test_logs_streaming():
    """Test the logs streaming endpoint"""
    client = TestClient(app)
    
    response = client.get("/api/v1/logs")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")

def test_invalid_request():
    """Test handling of invalid requests"""
    client = TestClient(app)
    
    # Test with invalid data
    response = client.post("/api/v1/run", json={"invalid": "data"})
    assert response.status_code == 422  # Unprocessable Entity
    
    data = response.json()
    assert "detail" in data

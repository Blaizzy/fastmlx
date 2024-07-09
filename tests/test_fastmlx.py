#!/usr/bin/env python

"""Tests for `fastmlx` package."""

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

# Import the actual classes and functions
from fastmlx import ChatCompletionRequest, ChatMessage, ModelProvider, app


# Create mock classes that inherit from the original classes
class MockModelProvider(ModelProvider):
    def __init__(self):
        super().__init__()
        self.models = {}

    def load_model(self, model_name: str):
        if model_name not in self.models:
            model_type = "vlm" if "llava" in model_name.lower() else "lm"
            self.models[model_name] = {
                "model": MagicMock(),
                "processor": MagicMock(),
                "tokenizer": MagicMock(),
                "image_processor": MagicMock() if model_type == "vlm" else None,
                "config": {"model_type": model_type},
            }
        return self.models[model_name]

    def get_available_models(self):
        return list(self.models.keys())


# Mock MODELS dictionary
MODELS = {"vlm": ["llava"], "lm": ["phi"]}


# Mock functions
def mock_generate(*args, **kwargs):
    return "generated response"


@pytest.fixture(scope="module")
def client():
    # Apply patches
    with patch("fastmlx.fastmlx.model_provider", MockModelProvider()), patch(
        "fastmlx.fastmlx.vlm_generate", mock_generate
    ), patch("fastmlx.fastmlx.lm_generate", mock_generate), patch(
        "fastmlx.fastmlx.MODELS", MODELS
    ):
        yield TestClient(app)


def test_chat_completion_vlm(client):
    request = ChatCompletionRequest(
        model="test_llava_model",
        messages=[ChatMessage(role="user", content="Hello")],
        image="test_image",
    )
    response = client.post(
        "/v1/chat/completions", json=json.loads(request.model_dump_json())
    )

    assert response.status_code == 200
    assert "generated response" in response.json()["choices"][0]["message"]["content"]


def test_chat_completion_lm(client):
    request = ChatCompletionRequest(
        model="test_phi_model", messages=[ChatMessage(role="user", content="Hello")]
    )
    response = client.post(
        "/v1/chat/completions", json=json.loads(request.model_dump_json())
    )

    assert response.status_code == 200
    assert "generated response" in response.json()["choices"][0]["message"]["content"]


def test_list_models(client):
    client.post("/v1/models?model_name=test_llava_model")
    client.post("/v1/models?model_name=test_phi_model")

    response = client.get("/v1/models")

    assert response.status_code == 200
    assert set(response.json()["models"]) == {"test_llava_model", "test_phi_model"}


def test_add_model(client):
    response = client.post("/v1/models?model_name=new_llava_model")

    assert response.status_code == 200
    assert response.json() == {
        "status": "success",
        "message": "Model new_llava_model added successfully",
    }


if __name__ == "__main__":
    pytest.main(["-v", __file__])

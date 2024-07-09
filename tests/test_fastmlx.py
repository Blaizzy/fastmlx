#!/usr/bin/env python

"""Tests for `fastmlx` package."""

import sys
from unittest.mock import MagicMock

# Detailed mock for mlx_vlm
class MockMLXVLM:
    class prompt_utils:
        @staticmethod
        def get_message_json(*args, **kwargs):
            return {"role": "user", "content": "mocked content"}

    class utils:
        @staticmethod
        def load_image_processor(*args, **kwargs):
            return MagicMock()

        @staticmethod
        def load_config(*args, **kwargs):
            return {"model_type": "test_model"}

    @staticmethod
    def load(*args, **kwargs):
        return (MagicMock(), MagicMock())

    @staticmethod
    def generate(*args, **kwargs):
        return "This is a test response."

# Detailed mock for mlx
class MockMLX:
    class core:
        @staticmethod
        def array(*args, **kwargs):
            return MagicMock()

        @staticmethod
        def concatenate(*args, **kwargs):
            return MagicMock()

    class nn:
        Module = type('Module', (), {})
        Linear = type('Linear', (), {'__call__': lambda *args, **kwargs: MagicMock()})

    @staticmethod
    def optimizer(*args, **kwargs):
        return MagicMock()

# Detailed mock for huggingface_hub
class MockHuggingFaceHub:
    class utils:
        class _errors:
            RepositoryNotFoundError = type('RepositoryNotFoundError', (Exception,), {})

    @staticmethod
    def snapshot_download(*args, **kwargs):
        return "/mocked/path/to/model"

# Apply mocks
sys.modules['mlx_vlm'] = MockMLXVLM()
sys.modules['mlx_vlm.prompt_utils'] = MockMLXVLM.prompt_utils
sys.modules['mlx_vlm.utils'] = MockMLXVLM.utils
sys.modules['mlx'] = MockMLX()
sys.modules['mlx.core'] = MockMLX.core
sys.modules['mlx.nn'] = MockMLX.nn
sys.modules['huggingface_hub'] = MockHuggingFaceHub()
sys.modules['huggingface_hub.utils'] = MockHuggingFaceHub.utils
sys.modules['huggingface_hub.utils._errors'] = MockHuggingFaceHub.utils._errors

from fastapi.testclient import TestClient
from fastapi import HTTPException
from unittest.mock import patch
from fastmlx import app

client = TestClient(app)

def test_chat_completion():
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "Hello"}],
            "max_tokens": 100,
            "temperature": 0.7
        }
    )
    assert response.status_code == 200
    assert "choices" in response.json()
    assert response.json()["choices"][0]["message"]["content"] == "This is a test response."

def test_chat_completion_with_image():
    response = client.post(
        "/v1/chat/completions",
        json={
            "model": "test-model",
            "messages": [{"role": "user", "content": "What's in this image?"}],
            "image": "base64_encoded_image_data",
            "max_tokens": 100,
            "temperature": 0.7
        }
    )
    assert response.status_code == 200
    assert "choices" in response.json()
    assert response.json()["choices"][0]["message"]["content"] == "This is a test response."

def test_list_models():
    with patch('fastmlx.ModelProvider.get_available_models', return_value=["model1", "model2"]):
        response = client.get("/v1/models")
    assert response.status_code == 200
    assert response.json() == {"models": ["model1", "model2"]}

def test_add_model():
    with patch('fastmlx.ModelProvider.add_model_path') as mock_add_model_path:
        response = client.post("/v1/models?model_name=new-model&model_path=/path/to/model")
    assert response.status_code == 200
    assert response.json() == {"status": "success", "message": "Model new-model added successfully"}
    mock_add_model_path.assert_called_once_with("new-model", "/path/to/model")


def test_chat_completion_invalid_model():
    with patch('fastmlx.ModelProvider.load_model', side_effect=HTTPException(status_code=404, detail="Model not found")):
        response = client.post(
            "/v1/chat/completions",
            json={
                "model": "invalid-model",
                "messages": [{"role": "user", "content": "Hello"}]
            }
        )

    assert response.status_code == 404
    assert "Model not found" in response.json()["detail"]
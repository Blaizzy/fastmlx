# FastMLX

[![image](https://img.shields.io/pypi/v/fastmlx.svg)](https://pypi.python.org/pypi/fastmlx)
[![image](https://img.shields.io/conda/vn/conda-forge/fastmlx.svg)](https://anaconda.org/conda-forge/fastmlx)
[![image](https://pyup.io/repos/github/Blaizzy/fastmlx/shield.svg)](https://pyup.io/repos/github/Blaizzy/fastmlx)

**FastMLX is a high performance production ready API to host MLX models, including Vision Language Models (VLMs) and Language Models (LMs).**

-   Free software: Apache Software License 2.0
-   Documentation: https://Blaizzy.github.io/fastmlx

## Features

- **OpenAI-compatible API**: Easily integrate with existing applications that use OpenAI's API.
- **Dynamic Model Loading**: Load MLX models on-the-fly or use pre-loaded models for better performance.
- **Support for Multiple Model Types**: Compatible with various MLX model architectures.
- **Image Processing Capabilities**: Handle both text and image inputs for versatile model interactions.
- **Efficient Resource Management**: Optimized for high-performance and scalability.
- **Error Handling**: Robust error management for production environments.
- **Customizable**: Easily extendable to accommodate specific use cases and model types.

## Usage

1. **Installation**

   ```bash
   pip install fastmlx
   ```

2. **Running the Server**

   Start the FastMLX server:
   ```bash
   fastmlx
   ```
   or

   ```bash
   uvicorn fastmlx:app --reload
   ```

3. **Making API Calls**

   Use the API similar to OpenAI's chat completions:

   **Vision Language Model**

   ```python
   import requests
   import json

   url = "http://localhost:8000/v1/chat/completions"
   headers = {"Content-Type": "application/json"}
   data = {
       "model": "mlx-community/nanoLLaVA-1.5-4bit",
       "image": "http://images.cocodataset.org/,val2017/000000039769.jpg",
       "messages": [{"role": "user", "content": "What are these"}],
       "max_tokens": 100
   }

   response = requests.post(url, headers=headers, data=json.dumps(data))
   print(response.json())
   ```
   **Language Model**
   ```python
   import requests
   import json

   url = "http://localhost:8000/v1/chat/completions"
   headers = {"Content-Type": "application/json"}
   data = {
       "model": "mlx-community/gemma-2-9b-it-4bit",
       "messages": [{"role": "user", "content": "What is the capital of France?"}],
       "max_tokens": 100
   }

   response = requests.post(url, headers=headers, data=json.dumps(data))
   print(response.json())
   ```

4. **Adding a New Model**

   You can add new models to the API:

   ```python
   import requests

   url = "http://localhost:8000/v1/models"
   params = {
       "model_name": "hf-repo-or-path",
   }

   response = requests.post(url, params=params)
   print(response.json())
   ```

5. **Listing Available Models**

   To see all available models:

   ```python
   import requests

   url = "http://localhost:8000/v1/models"
   response = requests.get(url)
   print(response.json())
   ```

For more detailed usage instructions and API documentation, please refer to the [full documentation](https://Blaizzy.github.io/fastmlx).
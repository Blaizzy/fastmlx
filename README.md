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
       "image": "http://images.cocodataset.org/val2017/000000039769.jpg",
       "messages": [{"role": "user", "content": "What are these"}],
       "max_tokens": 100
   }

   response = requests.post(url, headers=headers, data=json.dumps(data))
   print(response.json())
   ```

   With streaming:
   ```python
   import requests
   import json

   def process_sse_stream(url, headers, data):
      response = requests.post(url, headers=headers, json=data, stream=True)

      if response.status_code != 200:
         print(f"Error: Received status code {response.status_code}")
         print(response.text)
         return

      full_content = ""

      try:
         for line in response.iter_lines():
               if line:
                  line = line.decode('utf-8')
                  if line.startswith('data: '):
                     event_data = line[6:]  # Remove 'data: ' prefix
                     if event_data == '[DONE]':
                           print("\nStream finished. ✅")
                           break
                     try:
                           chunk_data = json.loads(event_data)
                           content = chunk_data['choices'][0]['delta']['content']
                           full_content += content
                           print(content, end='', flush=True)
                     except json.JSONDecodeError:
                           print(f"\nFailed to decode JSON: {event_data}")
                     except KeyError:
                           print(f"\nUnexpected data structure: {chunk_data}")

      except KeyboardInterrupt:
         print("\nStream interrupted by user.")
      except requests.exceptions.RequestException as e:
         print(f"\nAn error occurred: {e}")

   if __name__ == "__main__":
      url = "http://localhost:8000/v1/chat/completions"
      headers = {"Content-Type": "application/json"}
      data = {
         "model": "mlx-community/nanoLLaVA-1.5-4bit",
         "image": "http://images.cocodataset.org/val2017/000000039769.jpg",
         "messages": [{"role": "user", "content": "What are these?"}],
         "max_tokens": 500,
         "stream": True
      }
      process_sse_stream(url, headers, data)
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

   With streaming:
   ```python
   import requests
   import json

   def process_sse_stream(url, headers, data):
      response = requests.post(url, headers=headers, json=data, stream=True)

      if response.status_code != 200:
         print(f"Error: Received status code {response.status_code}")
         print(response.text)
         return

      full_content = ""

      try:
         for line in response.iter_lines():
               if line:
                  line = line.decode('utf-8')
                  if line.startswith('data: '):
                     event_data = line[6:]  # Remove 'data: ' prefix
                     if event_data == '[DONE]':
                           print("\nStream finished. ✅")
                           break
                     try:
                           chunk_data = json.loads(event_data)
                           content = chunk_data['choices'][0]['delta']['content']
                           full_content += content
                           print(content, end='', flush=True)
                     except json.JSONDecodeError:
                           print(f"\nFailed to decode JSON: {event_data}")
                     except KeyError:
                           print(f"\nUnexpected data structure: {chunk_data}")

      except KeyboardInterrupt:
         print("\nStream interrupted by user.")
      except requests.exceptions.RequestException as e:
         print(f"\nAn error occurred: {e}")

   if __name__ == "__main__":
      url = "http://localhost:8000/v1/chat/completions"
      headers = {"Content-Type": "application/json"}
      data = {
         "model": "mlx-community/gemma-2-9b-it-4bit",
         "messages": [{"role": "user", "content": "Hi, how are you?"}],
         "max_tokens": 500,
         "stream": True
      }
      process_sse_stream(url, headers, data)
   ```

4. **Listing Available Models**

   To see all vision and language models supported by MLX:

   ```python
   import requests

   url = "http://localhost:8000/v1/supported_models"
   response = requests.get(url)
   print(response.json())
   ```

5. **List Available Models**

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

6. **Listing Available Models**

   To see all available models:

   ```python
   import requests

   url = "http://localhost:8000/v1/models"
   response = requests.get(url)
   print(response.json())
   ```

7. **Delete Models**

   To remove any models loaded to memory:

   ```python
   import requests

   url = "http://localhost:8000/v1/models"
   params = {
      "model_name": "hf-repo-or-path",
   }
   response = requests.delete(url, params=params)
   print(response)
   ```

For more detailed usage instructions and API documentation, please refer to the [full documentation](https://Blaizzy.github.io/fastmlx).
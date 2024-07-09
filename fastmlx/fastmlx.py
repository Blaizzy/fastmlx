"""Main module."""
import os
import time
from typing import List, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import mlx.core as mx
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import get_message_json
from mlx_vlm.utils import load_image_processor, load_config

class ModelProvider:
    def __init__(self):
        self.models = {}


    def load_model(self, model_name: str):
        if model_name not in self.models:
            model, processor = load(model_name, {"trust_remote_code":True})
            image_processor = load_image_processor(model_name)
            config = load_config(model_name)
            self.models[model_name] = {
                "model": model,
                "processor": processor,
                "image_processor": image_processor,
                "config": config
            }

        return self.models[model_name]

    def add_model_path(self, model_name: str, model_path: str):
        self.model_paths[model_name] = model_path

    def get_available_models(self):
        return list(self.model_paths.keys())

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    image: Optional[str] = Field(default=None)
    max_tokens: Optional[int] = Field(default=100)
    temperature: Optional[float] = Field(default=0.7)

class ChatCompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[dict]

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the ModelProvider
model_provider = ModelProvider()

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completion(request: ChatCompletionRequest):
    model_data = model_provider.load_model(request.model)
    model = model_data["model"]
    processor = model_data["processor"]
    image_processor = model_data["image_processor"]
    config = model_data["config"]
    image = request.image

    chat_messages = []

    for msg in request.messages:
        if msg.role == "user":
            chat_messages.append(get_message_json(config["model_type"], msg.content))
        else:
            chat_messages.append({"role": msg.role, "content": msg.content})

    prompt = ""
    if "chat_template" in processor.__dict__.keys():
        prompt = processor.apply_chat_template(
            chat_messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    elif "tokenizer" in processor.__dict__.keys():
        if model.config.model_type != "paligemma":
            prompt = processor.tokenizer.apply_chat_template(
                chat_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = request.messages[-1].content


    # Generate the response
    output = generate(model, processor, image, prompt, image_processor, verbose=False)

    # Prepare the response
    response = ChatCompletionResponse(
        id=f"chatcmpl-{os.urandom(4).hex()}",
        created=int(time.time()),
        model=request.model,
        choices=[
            {
                "index": 0,
                "message": {"role": "assistant", "content": output},
                "finish_reason": "stop",
            }
        ],
    )

    return response

@app.get("/v1/models")
async def list_models():
    return {"models": model_provider.get_available_models()}

@app.post("/v1/models")
async def add_model(model_name: str, model_path: str):
    model_provider.add_model_path(model_name, model_path)
    return {"status": "success", "message": f"Model {model_name} added successfully"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
"""Main module."""
import os
import time
from typing import List, Optional
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

try:
    import mlx.core as mx
    from mlx_lm import generate as lm_generate
    from mlx_vlm import generate as vlm_generate
    from mlx_vlm.prompt_utils import get_message_json
    from mlx_vlm.utils import load_config
    from .utils import load_lm_model, load_vlm_model, MODEL_REMAPPING, MODELS
    MLX_AVAILABLE = True
except ImportError:
    print("Warning: mlx or mlx_lm not available. Some functionality will be limited.")
    MLX_AVAILABLE = False


class ModelProvider:
    def __init__(self):
        self.models = {}

    def load_model(self, model_name: str):
        if model_name not in self.models:
            config = load_config(model_name)
            model_type = MODEL_REMAPPING.get(config["model_type"], config["model_type"])
            if model_type in MODELS["vlm"]:
                self.models[model_name] = load_vlm_model(model_name, config)
            else:
                self.models[model_name] = load_lm_model(model_name, config)

        return self.models[model_name]


    def get_available_models(self):
        return list(self.models.keys())

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
    if not MLX_AVAILABLE:
        raise HTTPException(status_code=500, detail="MLX library not available")

    model_data = model_provider.load_model(request.model)
    model = model_data["model"]
    config = model_data["config"]
    model_type = MODEL_REMAPPING.get(config["model_type"], config["model_type"])

    if model_type in MODELS["vlm"]:
        processor = model_data["processor"]
        image_processor = model_data["image_processor"]

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
        output = vlm_generate(model, processor, image, prompt, image_processor, verbose=False)

    else:
        tokenizer = model_data["tokenizer"]
        chat_messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        if "chat_template" in tokenizer.__dict__.keys():
            prompt = tokenizer.apply_chat_template(
                chat_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = request.messages[-1].content

        output = lm_generate(model, tokenizer, prompt, verbose=False)


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
async def add_model(model_name: str):
    model_provider.load_model(model_name)
    return {"status": "success", "message": f"Model {model_name} added successfully"}

def run():
    import uvicorn
    uvicorn.run("fastmlx:app", host="127.0.0.1", port=8000, reload=True)

if __name__ == "__main__":
    run()
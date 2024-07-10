import json
import os
import time
from typing import Any, Dict, List, Optional

from pydantic import BaseModel

# MLX Imports
try:
    from mlx_lm import load as lm_load
    from mlx_lm import models as lm_models
    from mlx_lm.utils import stream_generate as lm_stream_generate
    from mlx_vlm import load as vlm_load
    from mlx_vlm import models as vlm_models
    from mlx_vlm.utils import (
        generate_step,
        load_image_processor,
        prepare_inputs,
        sample,
    )
except ImportError:
    print("Warning: mlx or mlx_lm not available. Some functionality will be limited.")


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: List[Dict[str, Any]]


def get_model_type_list(models, type="vlm"):

    # Get the directory path of the models package
    models_dir = os.path.dirname(models.__file__)

    # List all items in the models directory
    all_items = os.listdir(models_dir)

    if type == "vlm":
        submodules = [
            item
            for item in all_items
            if os.path.isdir(os.path.join(models_dir, item))
            and not item.startswith(".")
            and item != "__pycache__"
        ]
        return submodules
    else:
        return all_items


MODELS = {
    "vlm": get_model_type_list(vlm_models),
    "lm": get_model_type_list(lm_models, "lm"),
}
MODEL_REMAPPING = {"llava-qwen2": "llava_bunny", "bunny-llama": "llava_bunny"}


# Model Loading and Generation Functions
def load_vlm_model(model_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    model, processor = vlm_load(model_name, {"trust_remote_code": True})
    image_processor = load_image_processor(model_name)
    return {
        "model": model,
        "processor": processor,
        "image_processor": image_processor,
        "config": config,
    }


def load_lm_model(model_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    model, tokenizer = lm_load(model_name)
    return {"model": model, "tokenizer": tokenizer, "config": config}


# TODO: Replace on next mlx-vlm release
def vlm_stream_generate(
    model,
    processor,
    image: str,
    prompt: str,
    image_processor=None,
    max_tokens: int = 100,
    temp: float = 0.0,
    repetition_penalty: Optional[float] = None,
    repetition_context_size: Optional[int] = None,
    top_p: float = 1.0,
):

    if image_processor is not None:
        tokenizer = processor
    else:
        tokenizer = processor.tokenizer

    image_token_index = model.config.image_token_index
    input_ids, pixel_values, mask = prepare_inputs(
        image_processor, processor, image, prompt, image_token_index
    )
    logits, cache = model(input_ids, pixel_values, mask=mask)
    logits = logits[:, -1, :]
    y, _ = sample(logits, temp, top_p)

    detokenizer = processor.detokenizer
    detokenizer.reset()

    detokenizer.add_token(y.item())

    for (token, _), n in zip(
        generate_step(
            model.language_model,
            logits,
            mask,
            cache,
            temp,
            repetition_penalty,
            repetition_context_size,
            top_p,
        ),
        range(max_tokens),
    ):
        token = token.item()

        if token == tokenizer.eos_token_id:
            break

        detokenizer.add_token(token)
        detokenizer.finalize()
        yield detokenizer.last_segment


def vlm_stream_generator(
    model,
    model_name,
    processor,
    image,
    prompt,
    image_processor,
    max_tokens,
    temperature,
):
    for token in vlm_stream_generate(
        model,
        processor,
        image,
        prompt,
        image_processor,
        max_tokens=max_tokens,
        temp=temperature,
    ):
        chunk = ChatCompletionChunk(
            id=f"chatcmpl-{os.urandom(4).hex()}",
            created=int(time.time()),
            model=model_name,
            choices=[
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": token},
                    "finish_reason": None,
                }
            ],
        )
        yield f"data: {json.dumps(chunk.model_dump())}\n\n"
    yield "data: [DONE]\n\n"


def lm_stream_generator(model, model_name, tokenizer, prompt, max_tokens, temperature):
    for token in lm_stream_generate(
        model, tokenizer, prompt, max_tokens=max_tokens, temp=temperature
    ):
        chunk = ChatCompletionChunk(
            id=f"chatcmpl-{os.urandom(4).hex()}",
            created=int(time.time()),
            model=model_name,
            choices=[
                {
                    "index": 0,
                    "delta": {"role": "assistant", "content": token},
                    "finish_reason": None,
                }
            ],
        )
        yield f"data: {json.dumps(chunk.model_dump())}\n\n"
    yield "data: [DONE]\n\n"

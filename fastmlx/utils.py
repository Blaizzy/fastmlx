
from typing import Dict, Any
import os

# MLX Imports
try:
    from mlx_lm import load as lm_load, models as lm_models
    from mlx_vlm import load as vlm_load, models as vlm_models
    from mlx_vlm.utils import load_image_processor
except ImportError:
    print("Warning: mlx or mlx_lm not available. Some functionality will be limited.")


def get_model_type_list(models, type="vlm"):

    # Get the directory path of the models package
    models_dir = os.path.dirname(models.__file__)

    # List all items in the models directory
    all_items = os.listdir(models_dir)

    if type == "vlm":
        submodules = [item for item in all_items
              if os.path.isdir(os.path.join(models_dir, item))
              and not item.startswith('.')
              and item != '__pycache__']
        return submodules
    else:
        return all_items

MODELS = {"vlm": get_model_type_list(vlm_models), "lm": get_model_type_list(lm_models, "lm")}
MODEL_REMAPPING = {"llava-qwen2": "llava_bunny", "bunny-llama": "llava_bunny"}

# Model Loading and Generation Functions
def load_vlm_model(model_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    model, processor = vlm_load(model_name, {"trust_remote_code": True})
    image_processor = load_image_processor(model_name)
    return {
        "model": model,
        "processor": processor,
        "image_processor": image_processor,
        "config": config
    }

def load_lm_model(model_name: str, config: Dict[str, Any]) -> Dict[str, Any]:
    model, tokenizer = lm_load(model_name)
    return {
        "model": model,
        "tokenizer": tokenizer,
        "config": config
    }
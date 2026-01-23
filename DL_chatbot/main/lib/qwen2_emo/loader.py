# lib/qwen2_emo/loader.py

import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from huggingface_hub import snapshot_download

from .config import BASE_MODEL, ADAPTERS

_TOKENIZERS = {}
_MODELS = {}
removed = []

def _load_and_clean_adapter(adapter_repo: str) -> str:
    """
    Download adapter repo locally and remove invalid adapter fields
    Return local adapter directory
    """
    local_dir = snapshot_download(
        repo_id=adapter_repo,
        allow_patterns=[
            "adapter_config.json",
            "adapter_model.bin",
            "*.safetensors"
        ],
    )

    config_path = os.path.join(local_dir, "adapter_config.json")

    with open(config_path, "r", encoding="utf-8") as f:
        config_data = json.load(f)

    BAD_KEYS = [
    "alpha_pattern",
    "auto_mapping",
    "corda_config",
    "eva_config",
    "exclude_modules",
    "layer_replication",
    "layers_pattern",
    "layers_to_transform",
    "loftq_config",
    "megatron_config",
    "megatron_core",
    "qalora_group_size",
    "rank_pattern",
    "revision",
    "target_parameters",
    "trainable_token_indices",
    "use_dora",
    "use_qalora",
    "use_rslora",
    "lora_bias"
    ]


    # 🔥 核心修复：删除 PEFT 不支持的字段
    for bad_key in BAD_KEYS:
        if bad_key in config_data:
            del config_data[bad_key]
            removed.append(bad_key)

    if removed:
        print(f"[Adapter clean] removed keys: {removed}")

    # 只有真的修改过，才写回
    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config_data, f, indent=2)

    return local_dir

def load_model(model_type: str):
    """
    model_type: 'general' | 'emo'
    return: tokenizer, model (base + LoRA)
    """
    if model_type in _MODELS:
        return _TOKENIZERS[model_type], _MODELS[model_type]

    if model_type not in ADAPTERS:
        raise ValueError(f"Unknown model_type: {model_type}")

    # 1️⃣ tokenizer（来自 base model）
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL,
        trust_remote_code=True
    )

    # 2️⃣ base model
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True
    )

    # 3️⃣ download + clean adapter
    local_adapter_dir = _load_and_clean_adapter(
        ADAPTERS[model_type]
    )

    # 4️⃣ attach LoRA
    model = PeftModel.from_pretrained(
        base_model,
        local_adapter_dir
    )
    model.eval()

    _TOKENIZERS[model_type] = tokenizer
    _MODELS[model_type] = model

    return tokenizer, model

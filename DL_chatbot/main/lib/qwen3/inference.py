# lib/qwen3/inference.py
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# ✅ 改成你的模型路径：本地目录 or HF repo id
DEFAULT_QWEN3_MODEL = os.environ.get("QWEN3_MODEL", "Qwen/Qwen3-4B-Instruct-2507")


_MODEL = None
_TOKENIZER = None
_DEVICE = None


def load_qwen3(model_name_or_path: str = DEFAULT_QWEN3_MODEL,
               device: Optional[str] = None):
    global _MODEL, _TOKENIZER, _DEVICE

    if _MODEL is not None and _TOKENIZER is not None:
        return _MODEL, _TOKENIZER, _DEVICE

    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _TOKENIZER = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=False,
        use_fast=True,
    )

    if _DEVICE.type == "cuda":
        from transformers import BitsAndBytesConfig

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )

        _MODEL = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            trust_remote_code=False,
            quantization_config=bnb_config,
            device_map="auto",
            use_safetensors=True,
        )
    else:
        # CPU fallback（不建议跑 4B）
        _MODEL = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch.float32,
            use_safetensors=True,
        ).to(_DEVICE)

    _MODEL.eval()
    return _MODEL, _TOKENIZER, _DEVICE




HistoryType = Union[
    List[Dict[str, str]],                 # [{"role":"user","content":"..."}, ...]
    List[Tuple[str, str]],                # [(user, assistant), ...]  (来自 gr.Chatbot)
]


def _normalize_history(history: Optional[HistoryType]) -> List[Dict[str, str]]:
    """
    支持两种格式：
    1) list[dict]: {"role": "...", "content": "..."}
    2) list[tuple]: (user, assistant) from gr.Chatbot
    """
    if not history:
        return []

    norm: List[Dict[str, str]] = []

    # dict 格式
    if isinstance(history[0], dict):
        for item in history:  # type: ignore
            role = item.get("role", "")
            content = item.get("content", "")
            if role and content:
                norm.append({"role": role, "content": content})
        return norm

    # tuple 格式
    for u, a in history:  # type: ignore
        if u:
            norm.append({"role": "user", "content": str(u)})
        if a:
            norm.append({"role": "assistant", "content": str(a)})
    return norm


@torch.inference_mode()
def chat_qwen3(
    user_text: str,
    system_prompt: str = "You are an emotionally intelligent assistant. Respond with empathy and emotional awareness.",
    history: Optional[HistoryType] = None,
    model_name_or_path: str = DEFAULT_QWEN3_MODEL,
    max_new_tokens: int = 256,
    temperature: float = 0.7,
    top_p: float = 0.9,
    do_sample: bool = True,
) -> str:
    """
    ✅ 现在支持 history=
    """
    model, tokenizer, device = load_qwen3(model_name_or_path=model_name_or_path)

    messages: List[Dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    messages.extend(_normalize_history(history))
    messages.append({"role": "user", "content": user_text})

    # 使用 chat template（Qwen3 支持）
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    gen = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_p=top_p,
        pad_token_id=tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    # 只解码新生成部分
    input_len = inputs["input_ids"].shape[1]
    out_ids = gen[0][input_len:]
    return tokenizer.decode(out_ids, skip_special_tokens=True).strip()

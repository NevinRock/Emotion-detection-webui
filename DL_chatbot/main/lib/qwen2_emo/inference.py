# lib/qwen2_emo/inference.py

from .loader import load_model
from .config import SYSTEM_PROMPTS, DEFAULT_MODEL



def chat(
    text: str,
    history=None,
    model_type: str = DEFAULT_MODEL,
    max_new_tokens: int = 256,
):
    tokenizer, model = load_model(model_type)

    messages = []

    system_prompt = SYSTEM_PROMPTS.get(model_type)
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    if history:
        messages.extend(history)

    messages.append({"role": "user", "content": text})

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

    # ✅ 关键：只取“新生成的 tokens”，不要把 prompt（包含历史）decode出来
    prompt_len = inputs["input_ids"].shape[-1]
    gen_ids = outputs[0][prompt_len:]

    return tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

# lib/qwen2_emo/config.py

# ===== Base model（完整模型）=====
BASE_MODEL = "Qwen/Qwen2-1.5B-Instruct"

# ===== LoRA adapters（HF repo）=====
ADAPTERS = {
    "general": "ylhaichen04/Qwen2-1.5B-Instruct_LoRA_sft_general",
    "emo": "ylhaichen04/Qwen2-1.5B-Instruct_LoRA_sft_emo",
}

DEFAULT_MODEL = "general"

# ===== System prompts =====
SYSTEM_PROMPTS = {
    "general": "You are a helpful, concise assistant.",
    "emo": (
        "You are an emotionally intelligent assistant. "
        "Respond with empathy and emotional awareness."
    ),
}

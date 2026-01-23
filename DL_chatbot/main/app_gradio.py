import os
import socket
import gradio as gr
import torch
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# =========================
# HARD PATCH: gradio_client schema bool bug
# Fix both:
# - TypeError: argument of type 'bool' is not iterable
# - APIInfoParseError: Cannot parse schema True
# =========================
try:
    import gradio_client.utils as gcu  # type: ignore

    _old_json_schema_to_python_type = gcu._json_schema_to_python_type

    def _patched_json_schema_to_python_type(schema, defs=None):
        # JSON Schema spec allows boolean schemas: True / False
        if isinstance(schema, bool):
            return "Any"
        return _old_json_schema_to_python_type(schema, defs)

    gcu._json_schema_to_python_type = _patched_json_schema_to_python_type

    _old_get_type = gcu.get_type

    def _patched_get_type(schema):
        if isinstance(schema, bool):
            return "Any"
        return _old_get_type(schema)

    gcu.get_type = _patched_get_type

except Exception:
    pass


# =========================
# Local Imports
# =========================
import lib.CNN_ST as CNN_ST
import lib.ResNet50 as ResNet50
from lib.LSTM import LocalLSTMHandler
from lib.openai_api import gpt_response as openai_response
from lib.google_api import gpt_response as gemini_response
from lib.detect_face import FaceDetectorYOLO

# Qwen2 / Qwen3 lazy load
QWEN2_ERR = None
QWEN3_ERR = None
_qwen_chat = None
_chat_qwen3 = None


def _lazy_load_qwen2():
    global _qwen_chat, QWEN2_ERR
    if _qwen_chat is not None or QWEN2_ERR is not None:
        return
    try:
        from lib.qwen2_emo.inference import chat as qwen_chat
        _qwen_chat = qwen_chat
    except Exception as e:
        QWEN2_ERR = str(e)


def _lazy_load_qwen3():
    global _chat_qwen3, QWEN3_ERR
    if _chat_qwen3 is not None or QWEN3_ERR is not None:
        return
    try:
        from lib.qwen3.inference import chat_qwen3
        _chat_qwen3 = chat_qwen3
    except Exception as e:
        QWEN3_ERR = str(e)


# =========================
# Configuration & Paths
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT_DIR = os.path.join(BASE_DIR, "ckpt")

IMAGE_MODELS = {
    "ResNet50 (Facial)": {
        "path": os.path.join(CKPT_DIR, "ResNet50", "ckpt_facial", "ResNet50_facial.pth"),
        "handler": ResNet50
    },
    "ResNet50 (Human)": {
        "path": os.path.join(CKPT_DIR, "ResNet50", "ckpt_human", "ResNet50_human.pth"),
        "handler": ResNet50
    },
    "MyCNN (Facial)": {
        "path": os.path.join(CKPT_DIR, "CNN_ST", "ckpt_facial", "CNN_ST_facial.pth"),
        "handler": CNN_ST
    },
    "MyCNN (Human)": {
        "path": os.path.join(CKPT_DIR, "CNN_ST", "ckpt_human", "CNN_ST_human.pth"),
        "handler": CNN_ST
    }
}

WEIGHT_PATH_LSTM = os.path.join(CKPT_DIR, "LSTM", "ppl_234.pt")

# =========================
# Model Initialization
# =========================
lstm_bot = LocalLSTMHandler(model_path=WEIGHT_PATH_LSTM)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
face_detector = FaceDetectorYOLO(device=device)


def _chatbot_to_messages(chat_history):
    """
    Gradio Chatbot history: list[tuple(user, assistant)]
    -> list[dict] [{"role":"user","content":...}, {"role":"assistant","content":...}]
    """
    msgs = []
    if not chat_history:
        return msgs
    for user, assistant in chat_history:
        if user is not None and str(user).strip():
            msgs.append({"role": "user", "content": str(user)})
        if assistant is not None and str(assistant).strip():
            msgs.append({"role": "assistant", "content": str(assistant)})
    return msgs


def _build_qwen3_prompt(user_prompt: str, llm_history, max_turns: int = 12) -> str:
    """
    方案B：把历史拼进 prompt，模拟长上下文
    注意：Qwen3 的 chat_qwen3() 不再传 history=，避免 unexpected kwarg
    """
    if not llm_history:
        return user_prompt

    recent = llm_history[-max_turns:]
    lines = []
    for m in recent:
        role = m.get("role", "")
        content = m.get("content", "")
        if not content:
            continue
        if role == "user":
            lines.append(f"User: {content}")
        elif role == "assistant":
            lines.append(f"Assistant: {content}")
        else:
            lines.append(f"{role}: {content}")

    lines.append(f"User: {user_prompt}")
    lines.append("Assistant:")
    return "\n".join(lines)


# =====================================================
# Auto Explain
# =====================================================
def auto_explain_emotion(emotion_labels, history, model_mode):
    if history is None:
        history = []

    if not emotion_labels:
        history.append(("System", "Please upload an image and complete recognition first."))
        return history

    labels_to_explain = emotion_labels if isinstance(emotion_labels, list) else [emotion_labels]

    for label in labels_to_explain:
        prompt_text = (
            f"Detected '{label}'. Explain the meaning of this facial expression. "
            f"Be concise, point-form, and helpful."
        )

        try:
            llm_history = _chatbot_to_messages(history)

            if model_mode == "OpenAI":
                answer = openai_response(prompt_text, history=llm_history)

            elif model_mode == "Gemini":
                answer = gemini_response(prompt_text, history=llm_history)

            elif model_mode == "Custom LSTM":
                answer = lstm_bot.chat(user_text=prompt_text)

            elif model_mode == "Qwen2-General":
                _lazy_load_qwen2()
                if _qwen_chat is None:
                    answer = f"Qwen2 not available: {QWEN2_ERR}"
                else:
                    answer = _qwen_chat(prompt_text, history=llm_history, model_type="general")

            elif model_mode == "Qwen2-Emo":
                _lazy_load_qwen2()
                if _qwen_chat is None:
                    answer = f"Qwen2 not available: {QWEN2_ERR}"
                else:
                    answer = _qwen_chat(prompt_text, history=llm_history, model_type="emo")

            elif model_mode == "Qwen3":
                _lazy_load_qwen3()
                if _chat_qwen3 is None:
                    answer = f"Qwen3 not available: {QWEN3_ERR}"
                else:
                    qwen3_prompt = _build_qwen3_prompt(prompt_text, llm_history)
                    # ✅ 不传 history=
                    try:
                        answer = _chat_qwen3(
                            qwen3_prompt,
                            system_prompt="You are an emotionally intelligent assistant. Respond with empathy and emotional awareness.",
                        )
                    except TypeError:
                        # 如果你的 chat_qwen3 根本不支持 system_prompt，就降级
                        answer = _chat_qwen3(qwen3_prompt)

            else:
                answer = f"Unknown model selected: {model_mode}"

            history.append((f"Explain '{label}'", answer))

        except Exception as e:
            history.append((f"Explain '{label}'", str(e)))

    return history


# =====================================================
# Chat Follow-up
# =====================================================
def chat_followup(user_text, emotion_label, history, model_mode):
    if history is None:
        history = []

    if not user_text or not user_text.strip():
        return history, ""

    if not emotion_label:
        history.append((user_text, "Please upload an image first so I know which expression we're discussing."))
        return history, ""

    try:
        llm_history = _chatbot_to_messages(history)

        if model_mode == "OpenAI":
            prompt = (
                f"Recognized emotion(s): {emotion_label}\n"
                f"User input: {user_text}\n"
                f"Respond in English, helpful and concise."
            )
            response_text = openai_response(prompt, history=llm_history)

        elif model_mode == "Gemini":
            prompt = (
                f"Recognized emotion(s): {emotion_label}\n"
                f"User input: {user_text}\n"
                f"Respond in English, helpful and concise."
            )
            response_text = gemini_response(prompt, history=llm_history)

        elif model_mode == "Custom LSTM":
            response_text = lstm_bot.chat(user_text=user_text)

        elif model_mode == "Qwen2-General":
            _lazy_load_qwen2()
            if _qwen_chat is None:
                response_text = f"Qwen2 not available: {QWEN2_ERR}"
            else:
                prompt_with_context = f"Current detected emotions: {emotion_label}\nUser Question: {user_text}"
                response_text = _qwen_chat(prompt_with_context, history=llm_history, model_type="general")

        elif model_mode == "Qwen2-Emo":
            _lazy_load_qwen2()
            if _qwen_chat is None:
                response_text = f"Qwen2 not available: {QWEN2_ERR}"
            else:
                prompt_with_context = f"Current detected emotions: {emotion_label}\nUser Question: {user_text}"
                response_text = _qwen_chat(prompt_with_context, history=llm_history, model_type="emo")

        elif model_mode == "Qwen3":
            _lazy_load_qwen3()
            if _chat_qwen3 is None:
                response_text = f"Qwen3 not available: {QWEN3_ERR}"
            else:
                prompt_with_context = f"Detected emotion(s): {emotion_label}\nUser says: {user_text}"
                qwen3_prompt = _build_qwen3_prompt(prompt_with_context, llm_history)
                # ✅ 不传 history=
                try:
                    response_text = _chat_qwen3(
                        qwen3_prompt,
                        system_prompt="You are an emotionally intelligent assistant. Respond with empathy and emotional awareness.",
                    )
                except TypeError:
                    response_text = _chat_qwen3(qwen3_prompt)

        else:
            response_text = f"Unknown model selected: {model_mode}"

        history.append((user_text, response_text))

    except Exception as e:
        history.append((user_text, str(e)))

    return history, ""


# =====================================================
# Predict Emotion
# =====================================================
def predict_emotion(input_img, model_choice):
    if input_img is None:
        return None, "Please drag or upload an image", "", None

    face_boxes = face_detector.detect(input_img)
    if len(face_boxes) == 0:
        return input_img, "No face detected", "", None

    annotated_img = input_img.copy()
    draw = ImageDraw.Draw(annotated_img)

    try:
        font = ImageFont.truetype("arial.ttf", 24)
    except IOError:
        font = ImageFont.load_default()

    labels = []
    probs_text = []

    for i, (x1, y1, x2, y2) in enumerate(face_boxes):
        face_crop = input_img.crop((x1, y1, x2, y2))
        temp_face_path = f"temp_face_{i}.jpg"
        face_crop.save(temp_face_path)

        try:
            selected_model = IMAGE_MODELS[model_choice]
            handler = selected_model["handler"]
            weight_path = selected_model["path"]

            label, probs = handler.classify_emotion(temp_face_path, weight_path)

            face_id = f"Face {i+1}"
            labels.append(f"{face_id}: {label}")

            prob_str = f"{face_id} Probabilities:\n"
            for k, v in probs.items():
                prob_str += f"{k:<10}: {v:.2f}%\n"
            probs_text.append(prob_str)

            draw.rectangle([x1, y1, x2, y2], outline="red", width=3)

            text_caption = f"{i+1}. {label}"
            text_bbox = draw.textbbox((x1, y1), text_caption, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]

            text_x = x1
            text_y = y1 - text_height - 5 if y1 > (text_height + 5) else y1

            draw.rectangle(
                [text_x, text_y, text_x + text_width + 4, text_y + text_height + 4],
                fill="red"
            )
            draw.text((text_x + 2, text_y), text_caption, fill="white", font=font)

        except Exception as e:
            labels.append(f"Face {i+1}: Error")
            probs_text.append(str(e))
        finally:
            if os.path.exists(temp_face_path):
                os.remove(temp_face_path)

    return annotated_img, "\n".join(labels), "\n\n".join(probs_text), labels


# =====================================================
# UI
# =====================================================
with gr.Blocks(title="Emotion Recognition") as demo:
    gr.Markdown("# Facial Expression Recognition Web UI")
    gr.Markdown("Upload an image. The system will detect faces, recognize emotions, and label them.")

    emotion_state = gr.State(None)

    with gr.Row():
        image_model_selector = gr.Dropdown(
            choices=list(IMAGE_MODELS.keys()),
            value="ResNet50 (Facial)",
            label="Image Classification Model",
            interactive=True
        )

    with gr.Row():
        with gr.Column(scale=1):
            input_img = gr.Image(type="pil", label="Original Input", sources=["upload", "clipboard"])
        with gr.Column(scale=1):
            output_img = gr.Image(type="pil", label="Detected Result (Face ID)")

    with gr.Row():
        output_label = gr.Textbox(label="Prediction Summary", lines=3)
        output_probs = gr.Textbox(label="Detailed Probabilities", lines=6)

    gr.Markdown("---")
    gr.Markdown("### AI Assistant (Chat about results)")

    with gr.Row():
        model_selector = gr.Radio(
            choices=["OpenAI", "Gemini", "Custom LSTM", "Qwen2-General", "Qwen2-Emo", "Qwen3"],
            value="Qwen2-Emo",
            label="Chat Model"
        )

    chat = gr.Chatbot(height=350, label="Conversation")

    with gr.Row():
        user_box = gr.Textbox(label="Your Message", placeholder="Ask about the emotion...", scale=4)
        send_btn = gr.Button("Send", scale=1, variant="primary")

    clear_btn = gr.Button("Clear Chat")

    input_img.change(
        fn=predict_emotion,
        inputs=[input_img, image_model_selector],
        outputs=[output_img, output_label, output_probs, emotion_state]
    )

    image_model_selector.change(
        fn=predict_emotion,
        inputs=[input_img, image_model_selector],
        outputs=[output_img, output_label, output_probs, emotion_state]
    )

    emotion_state.change(
        fn=auto_explain_emotion,
        inputs=[emotion_state, chat, model_selector],
        outputs=[chat]
    )

    send_btn.click(
        fn=chat_followup,
        inputs=[user_box, emotion_state, chat, model_selector],
        outputs=[chat, user_box]
    )

    user_box.submit(
        fn=chat_followup,
        inputs=[user_box, emotion_state, chat, model_selector],
        outputs=[chat, user_box]
    )

    clear_btn.click(lambda: [], None, chat)


def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


if __name__ == "__main__":
    port = get_free_port()
    print(f"Starting server on port: {port}")
    demo.queue()
    demo.launch(share=True, server_port=port, show_api=False)

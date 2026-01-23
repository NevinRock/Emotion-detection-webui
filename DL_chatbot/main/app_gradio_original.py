import gradio as gr
import os
import socket
import torch
from lib.qwen2_emo.inference import chat as qwen_chat


# =========================
# 原有 imports（不变）
# =========================
import lib.CNN_ST as CNN_ST 
import lib.ResNet50 as ResNet50
from lib.LSTM import LocalLSTMHandler
from lib.openai_api import gpt_response

# =========================
# 【新增】YOLO 人脸检测
# =========================
from lib.detect_face import FaceDetectorYOLO


# =========================
# 原有权重路径（不变）
# =========================
WEIGHT_PATH_CNN_ST = r"../ckpt/ResNet50/ckpt_facial/ResNet50_facial.pth"
WEIGHT_PATH_RESNET50 = r"../ckpt/ResNet50/ckpt_facial/ResNet50_facial.pth"
WEIGHT_PATH_LSTM = r"../ckpt/LSTM/ppl_234.pt"

# =========================
# 原有模型初始化（不变）
# =========================
lstm_bot = LocalLSTMHandler(model_path=WEIGHT_PATH_LSTM)

# 【新增】YOLO detector
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
face_detector = FaceDetectorYOLO(
    device=device
)

# =====================================================
# 原有函数：auto_explain_emotion（⚠️只支持单标签）
# 👉 这里做【最小修改】：支持 list[str]
# =====================================================
def auto_explain_emotion(emotion_labels, history):
    if history is None:
        history = []

    if not emotion_labels:
        history.append((
            "System",
            "Please upload an image and complete recognition first."
        ))
        return history

    # 多脸
    if isinstance(emotion_labels, list):
        for label in emotion_labels:
            try:
                answer = gpt_response(
                    f"Explain the meaning of the facial expression '{label}'. "
                    f"Be concise, point-form, and helpful."
                )
                history.append((f"Explain '{label}'", answer))
            except Exception as e:
                history.append((f"Explain '{label}'", str(e)))
        return history

    # 单脸
    try:
        answer = gpt_response(
            f"Explain the meaning of the facial expression '{emotion_labels}'. "
            f"Be concise, point-form, and helpful."
        )
        history.append((f"Explain '{emotion_labels}'", answer))
    except Exception as e:
        history.append((f"Explain '{emotion_labels}'", str(e)))

    return history


# =====================================================
# 原有 chat_followup（完全不改）
# =====================================================
def chat_followup(user_text, emotion_label, history, model_mode):
    if history is None:
        history = []

    if not user_text or not user_text.strip():
        return history, ""

    if not emotion_label:
        history.append((
            user_text,
            "Please upload an image first so I know which expression we're discussing."
        ))
        return history, ""

    try:
        # ===== OpenAI =====
        if model_mode == "OpenAI":
            prompt = (
                f"Recognized emotion(s): {emotion_label}\n"
                f"User input: {user_text}\n"
                f"Respond in English, helpful and concise."
            )
            response_text = gpt_response(prompt)

        # ===== Custom LSTM =====
        elif model_mode == "Custom LSTM":
            response_text = lstm_bot.chat(user_text=user_text)

        # ===== Qwen2 General =====
        elif model_mode == "Qwen2-General":
            response_text = qwen_chat(
                user_text,
                history=None,        # UI history ≠ LLM history
                model_type="general"
            )

        # ===== Qwen2 Emo =====
        elif model_mode == "Qwen2-Emo":
            response_text = qwen_chat(
                user_text,
                history=None,
                model_type="emo"
            )

        # ===== Fallback =====
        else:
            response_text = f"Unknown model selected: {model_mode}"

        # ✅ Gradio Chatbot 正确格式
        history.append((user_text, response_text))

    except Exception as e:
        history.append((user_text, str(e)))

    return history, ""


# =====================================================
# 【核心修改】predict_emotion：YOLO → 多脸 → CNN
# =====================================================
def predict_emotion(input_img):
    if input_img is None:
        return "Please drag or upload an image", "", None

    # 1️⃣ YOLO 检测人脸
    face_boxes = face_detector.detect(input_img)

    if len(face_boxes) == 0:
        return "No face detected", "", None

    labels = []
    probs_text = []

    for i, (x1, y1, x2, y2) in enumerate(face_boxes):
        face_crop = input_img.crop((x1, y1, x2, y2))
        temp_face_path = f"temp_face_{i}.jpg"
        face_crop.save(temp_face_path)

        try:
            label, probs = ResNet50.classify_emotion(
                temp_face_path,
                WEIGHT_PATH_RESNET50
            )

            labels.append(f"Face {i+1}: {label}")

            prob_str = f"Face {i+1} Probabilities:\n"
            for k, v in probs.items():
                prob_str += f"{k:<10}: {v:.2f}%\n"
            probs_text.append(prob_str)

        finally:
            if os.path.exists(temp_face_path):
                os.remove(temp_face_path)

    return (
        "\n".join(labels),          # Prediction Result
        "\n\n".join(probs_text),    # Detailed Probabilities
        labels                      # emotion_state（list）
    )


# =====================================================
# Gradio UI（完全不改）
# =====================================================
with gr.Blocks() as demo:
    gr.Markdown("# 🚀 Facial Expression Recognition Web UI")
    gr.Markdown("Upload an image → predict expression → auto explanation appears below.")

    emotion_state = gr.State(None)

    with gr.Row():
        input_img = gr.Image(type="pil", label="Upload Image", width=400)
        with gr.Column():
            output_label = gr.Textbox(label="Prediction Result", lines=2)
            output_probs = gr.Textbox(label="Detailed Probabilities", lines=10)

    model_selector = gr.Radio(
        choices=["OpenAI", "Custom LSTM","Qwen2-General","Qwen2-Emo"], 
        value="OpenAI", 
        label="Select Chat Model"
    )

    chat = gr.Chatbot(height=400)

    with gr.Row():
        user_box = gr.Textbox(label="Your Message", scale=4)
        send_btn = gr.Button("Send", scale=1)

    clear_btn = gr.Button("Clear Chat")

    input_img.change(
        fn=predict_emotion,
        inputs=input_img,
        outputs=[output_label, output_probs, emotion_state]
    )

    emotion_state.change(
        fn=auto_explain_emotion,
        inputs=[emotion_state, chat],
        outputs=chat
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


# =====================================================
# Launch（不改）
# =====================================================
def get_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

demo.launch(share=True, server_port=get_free_port())

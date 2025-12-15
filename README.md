# COMP0220 Coursework-emotion detection 

## Introduction

This project follows a two-stage pipeline: a YOLO-based face detection module first extracts facial regions from the input image, and a ResNet classifier then performs fine-grained facial emotion recognition. The predicted emotion is used as an affective context for the language model, enabling more empathetic and human-centered conversations. To better align the chatbot with supportive and caring dialogue scenarios, we fine-tune the language model with LoRA, improving its consistency and appropriateness in emotional support responses.

Code organization: the main Web UI code is in `DL_chatbox/`, and the image-classification training/evaluation code is in `ImgClassification/`.

## Quick Setup

Using conda:

conda create -n webui_py39 python=3.9.25

conda acitivate webui_py39

pip install torch==2.4.1+cu118 torchvision==0.19.1+cu118 torchaudio==2.4.1+cu118 --index-url https://download.pytorch.org/whl/cu118

pip install -r requirement.txt

## Weight Download

The LLM weights are loaded directly from Hugging Face and therefore do not require manual downloading. All other model weights are already included in this repository.
### LLM weights: 
https://huggingface.co/ylhaichen04/Qwen2-1.5B-Instruct_LoRA_sft_general
https://huggingface.co/ylhaichen04/Qwen2-1.5B-Instruct_LoRA_sft_emo
## Dataset

### Image Classifier:

Human Face Emotions: https://www.kaggle.com/datasets/samithsachidanandan/human-face-emotions

Facial Emotion Recognition Dataset： https://www.kaggle.com/datasets/fahadullaha/facial-emotion-recognition-dataset

### LLM fine-turning

Empathetic Dialogues: https://huggingface.co/datasets/facebook/empathetic_dialogues

Alpaca: https://huggingface.co/datasets/tatsu-lab/alpaca

### LSTM 

Empathetic Dialogues: https://huggingface.co/datasets/facebook/empathetic_dialogues



## File Detail

Image classification training: `ImgClassification/`

webui： `DL_chatbox/`

LSTM training: `LSTM/`


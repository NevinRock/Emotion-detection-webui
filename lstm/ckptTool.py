import torch
# Import variables from your original training code (assume filename is train.py)
from train import model, word_to_idx, MAX_LEN, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT, MODEL_CHECKPOINT_PATH

# Load old weights
print("Loading old weights...")
checkpoint = torch.load(MODEL_CHECKPOINT_PATH, map_location='cpu')

# Create new dictionary
new_state = {
    "model_state_dict": checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint,
    "word_to_idx": word_to_idx,  # Store current vocab in memory
    "max_len": MAX_LEN,
    "config": {
        "emb_dim": EMBEDDING_DIM,
        "hid_dim": HIDDEN_DIM,
        "n_layers": NUM_LAYERS,
        "dropout": DROPOUT
    }
}

# Save as new file
output_path = "lstm_inference_ready.pt"
torch.save(new_state, output_path)
print(f"Conversion complete! Please use {output_path} for Gradio inference.")
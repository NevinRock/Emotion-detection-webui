import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import pandas as pd
import os
import re
import random
import time
import math
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import CountVectorizer

# ==============================================================================
# 0. Config & File Paths
# ==============================================================================

# Embedding dimension must match GloVe file
EMBEDDING_DIM = 100          # Matches glove.6B.100d.txt
MAX_FEATURES = 20000

# Model Architecture
MAX_LEN = 50                 # Maximum sentence length
BATCH_SIZE = 256
HIDDEN_DIM = 128
NUM_LAYERS = 2
DROPOUT = 0.5               # Don't be too large, 0.5 is more stable than 0.7
N_EPOCHS = 10
CLIP = 1.0
TEACHER_FORCING_RATIO = 0  # Can be changed to decay with epoch later

# Special Tokens
PAD_TOKEN = '<pad>'
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
UNK_TOKEN = '<unk>'
SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

# File Paths
TRAIN_CSV_PATH = r"C:\Users\Harry\Desktop\DeepLearning_RAI\Dialogue_CW\empathetic_dialog_datasets\processed\train_processed.csv"
VALID_CSV_PATH = r"C:\Users\Harry\Desktop\DeepLearning_RAI\Dialogue_CW\empathetic_dialog_datasets\processed\valid_processed.csv"
PRETRAINED_EMBEDDINGS_PATH = r"C:\Users\Harry\Desktop\DeepLearning_RAI\Dialogue_CW\glove.6B\glove.6B.100d.txt"

MODEL_CHECKPOINT_PATH = "best_empathetic_lstm_attn_glove.pt"

SAMPLE_FRACTION = 1.0        # Can be changed to 0.3 / 0.5 etc. for speedup

# Device & Random Seed
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device used: {DEVICE}")

SCALER = torch.amp.GradScaler(enabled=(DEVICE.type == "cuda"))

# Emoji + Tokenization
EMOJI_PATTERN = re.compile(
    "[" 
    "\U0001F600-\U0001F64F"  # Emoticons
    "\U0001F300-\U0001F5FF"  # Symbols & Pictographs
    "\U0001F680-\U0001F6FF"  # Transport & Map Symbols
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U00002600-\U000027B0"  # Miscellaneous Symbols
    "\\ufe0f"                # Variation Selector 16
    "]+",
    flags=re.UNICODE
)

def tokenize(text):
    """
    Tokenizer:
    1. Remove Emoji
    2. Lowercase
    3. Separate words and punctuation using regex, keeping abbreviations (don't, you're)
    """
    text = str(text)
    text = EMOJI_PATTERN.sub(r"", text)
    text = text.lower()
    tokens = re.findall(r"[\w']+|[^\w\s]", text)
    return [t for t in tokens if t.strip()]

# ==============================================================================
# 1. Vocab + GloVe Loading
# ==============================================================================

def create_vocab_and_load_data(train_path, valid_path, embed_path):
    try:
        train_df = pd.read_csv(train_path)
        valid_df = pd.read_csv(valid_path)
    except FileNotFoundError:
        print("Error: Train or Valid CSV file not found, please check path.")
        exit()
    except Exception as e:
        print(f"Error reading CSV: {e}")
        exit()

    # Column Renaming
    train_df = train_df.rename(columns={"input": "input_text", "output": "target_text"})
    valid_df = valid_df.rename(columns={"input": "input_text", "output": "target_text"})

    if "input_text" not in train_df.columns or "target_text" not in train_df.columns:
        print("input_text / target_text columns missing in CSV, please check.")
        print("train_df columns:", train_df.columns)
        exit()

    print(f"Original Train Size: {len(train_df)}, Valid Size: {len(valid_df)}")

    # Optional: Downsampling
    if SAMPLE_FRACTION < 1.0:
        train_df = train_df.sample(frac=SAMPLE_FRACTION, random_state=SEED).reset_index(drop=True)
        valid_df = valid_df.sample(frac=SAMPLE_FRACTION, random_state=SEED).reset_index(drop=True)
        print(f"Reduced Train Size: {len(train_df)}, Valid Size: {len(valid_df)}")

    # Build Vocab
    all_texts = pd.concat([train_df["input_text"], train_df["target_text"]]).astype(str)
    vectorizer = CountVectorizer(tokenizer=tokenize, max_features=MAX_FEATURES, min_df=5)
    vectorizer.fit(all_texts)

    word_to_idx = {word: idx + len(SPECIAL_TOKENS) for word, idx in vectorizer.vocabulary_.items()}
    word_to_idx[PAD_TOKEN] = 0
    word_to_idx[SOS_TOKEN] = 1
    word_to_idx[EOS_TOKEN] = 2
    word_to_idx[UNK_TOKEN] = 3

    vocab_size = len(word_to_idx)
    print(f"Vocab Size (VOCAB_SIZE): {vocab_size}")

    # Load GloVe
    weights_matrix, found = load_pretrained_embeddings(embed_path, word_to_idx, EMBEDDING_DIM)

    print(f"GloVe Loading Completed: {found} words matched.")
    print(f"{vocab_size - found} words not found in GloVe, using zero vectors.")

    return train_df, valid_df, word_to_idx, vocab_size, weights_matrix

def load_pretrained_embeddings(path, w2i, emb_dim):
    print(f"Target Vocab Size: {len(w2i)}, Target Embedding Dimension: {emb_dim}")
    weights_matrix = np.zeros((len(w2i), emb_dim), dtype="float32")
    words_found = 0

    if not os.path.exists(path):
        print(f"Error: Pretrained file {path} not found")
        return torch.from_numpy(weights_matrix), 0

    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) <= emb_dim:
                    continue
                word = parts[0]
                if word in w2i:
                    idx = w2i[word]
                    try:
                        vector = np.asarray(parts[1:], dtype="float32")
                        if vector.shape[0] == emb_dim:
                            weights_matrix[idx] = vector
                            words_found += 1
                    except ValueError:
                        continue
    except Exception as e:
        print(f"Error loading GloVe: {e}")

    return torch.from_numpy(weights_matrix), words_found

# Run Vocab + GloVe Loading
train_df, valid_df, word_to_idx, VOCAB_SIZE, pretrained_weights = create_vocab_and_load_data(
    TRAIN_CSV_PATH, VALID_CSV_PATH, PRETRAINED_EMBEDDINGS_PATH
)
idx_to_word = {idx: word for word, idx in word_to_idx.items()}
TRG_PAD_IDX = word_to_idx[PAD_TOKEN]

# ==============================================================================
# 2. Dataset & DataLoader
# ==============================================================================

class DialogueDataset(Dataset):
    def __init__(self, df, w2i, max_len):
        self.input_sequences = df["input_text"].tolist()
        self.target_sequences = df["target_text"].tolist()
        self.w2i = w2i
        self.max_len = max_len

    def __len__(self):
        return len(self.input_sequences)

    def _text_to_sequence(self, text):
        if pd.isna(text):
            seq = [self.w2i[SOS_TOKEN], self.w2i[EOS_TOKEN]]
            seq += [self.w2i[PAD_TOKEN]] * (self.max_len - len(seq))
            return torch.tensor(seq, dtype=torch.long)
        tokens = tokenize(str(text))
        seq = [self.w2i[SOS_TOKEN]] + [self.w2i.get(t, self.w2i[UNK_TOKEN]) for t in tokens] + [self.w2i[EOS_TOKEN]]
        if len(seq) > self.max_len:
            seq = seq[:self.max_len]
        elif len(seq) < self.max_len:
            seq += [self.w2i[PAD_TOKEN]] * (self.max_len - len(seq))
        return torch.tensor(seq, dtype=torch.long)

    def __getitem__(self, idx):
        src = self._text_to_sequence(self.input_sequences[idx])
        trg = self._text_to_sequence(self.target_sequences[idx])
        return src, trg

train_dataset = DialogueDataset(train_df, word_to_idx, MAX_LEN)
valid_dataset = DialogueDataset(valid_df, word_to_idx, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)

# ==============================================================================
# 3. Encoder / Decoder / Seq2Seq with Attention + GloVe
# ==============================================================================

class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, n_layers, dropout, weights_matrix=None):
        super().__init__()
        if weights_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(
                weights_matrix, freeze=False, padding_idx=TRG_PAD_IDX
            )
        else:
            self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=TRG_PAD_IDX)

        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src: [batch, src_len]
        embedded = self.dropout(self.embedding(src))        # [batch, src_len, emb_dim]
        outputs, (hidden, cell) = self.rnn(embedded)        # outputs: [batch, src_len, hid_dim]
        return outputs, hidden, cell

class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, n_layers, dropout, weights_matrix=None):
        super().__init__()
        self.vocab_size = vocab_size
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        if weights_matrix is not None:
            self.embedding = nn.Embedding.from_pretrained(
                weights_matrix, freeze=False, padding_idx=TRG_PAD_IDX
            )
        else:
            self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=TRG_PAD_IDX)

        # Bahdanau attention
        self.attn = nn.Linear(hid_dim * 2, hid_dim)
        self.v = nn.Linear(hid_dim, 1, bias=False)

        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)

        self.fc_out = nn.Linear(hid_dim * 2, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell, encoder_outputs):
        """
        input: [batch] or [batch, 1]
        hidden: [n_layers, batch, hid_dim]
        cell:   [n_layers, batch, hid_dim]
        encoder_outputs: [batch, src_len, hid_dim]
        """
        if input.dim() == 1:
            input = input.unsqueeze(1)  # [batch, 1]

        embedded = self.dropout(self.embedding(input))      # [batch, 1, emb_dim]

        # 1. Decoder LSTM step
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))  # output: [batch, 1, hid_dim]

        # 2. Attention: Use top hidden as query
        hidden_top = hidden[-1]                                # [batch, hid_dim]
        src_len = encoder_outputs.size(1)

        hidden_expanded = hidden_top.unsqueeze(1).repeat(1, src_len, 1)  # [batch, src_len, hid_dim]
        energy_input = torch.cat((hidden_expanded, encoder_outputs), dim=2)  # [batch, src_len, 2*hid_dim]

        energy = torch.tanh(self.attn(energy_input))           # [batch, src_len, hid_dim]
        attention_scores = self.v(energy).squeeze(2)           # [batch, src_len]
        attention_weights = F.softmax(attention_scores, dim=1) # [batch, src_len]

        # 3. Weighted sum to get context
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs)  # [batch, 1, hid_dim]
        context = context.squeeze(1)                                          # [batch, hid_dim]
        output_squeezed = output.squeeze(1)                                   # [batch, hid_dim]

        concat = torch.cat((output_squeezed, context), dim=1)   # [batch, 2*hid_dim]
        prediction = self.fc_out(concat)                        # [batch, vocab_size]

        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        """
        src: [batch, src_len]
        trg: [batch, trg_len]
        """
        batch_size = trg.size(0)
        trg_len = trg.size(1)
        vocab_size = self.decoder.vocab_size

        outputs = torch.zeros(batch_size, trg_len, vocab_size, device=self.device)

        encoder_outputs, hidden, cell = self.encoder(src)
        input_token = trg[:, 0]  # First input is <sos>

        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input_token, hidden, cell, encoder_outputs)
            outputs[:, t, :] = output

            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)

            input_token = trg[:, t] if teacher_force else top1

        return outputs

# Instantiate Model (Integrate GloVe Weights)
enc = Encoder(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT, weights_matrix=pretrained_weights)
dec = Decoder(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT, weights_matrix=pretrained_weights)
model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)

criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# ==============================================================================
# 4. Training / Eval Functions
# ==============================================================================

def save_checkpoint(model, optimizer, epoch, best_valid_loss, filename):
    state = {
        "epoch": epoch,
        "best_valid_loss": best_valid_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        
        # --- New Key Part: Save Config and Vocab ---
        # So inference doesn't need CSV files
        "word_to_idx": word_to_idx,  # Ensure word_to_idx is global
        "max_len": MAX_LEN,
        "config": {
            "emb_dim": EMBEDDING_DIM,
            "hid_dim": HIDDEN_DIM,
            "n_layers": NUM_LAYERS,
            "dropout": DROPOUT
        }
    }
    torch.save(state, filename)
    print(f"Checkpointed to {filename} (with vocab included).")

def load_checkpoint(model, optimizer, filename):
    if not os.path.exists(filename):
        print("==> No checkpoint found, starting training from Epoch 1.")
        return 0, float("inf")

    print(f"==> Checkpoint found {filename}, loading...")
    # Add weights_only=False to allow loading old weights with Numpy data or complex dicts
    checkpoint = torch.load(MODEL_CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)

    if "vocab_size" not in checkpoint or "max_len" not in checkpoint:
        print("Checkpoint format old, only loading model parameters, training from scratch.")
        model.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
        return 0, float("inf")

    if checkpoint["vocab_size"] != VOCAB_SIZE or checkpoint["max_len"] != MAX_LEN:
        print("Checkpoint config mismatch, training from scratch.")
        return 0, float("inf")

    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    start_epoch = checkpoint["epoch"] + 1
    best_valid_loss = checkpoint["best_valid_loss"]
    print(f"==> Loaded successfully, starting from Epoch {start_epoch}, Current Best Valid Loss: {best_valid_loss:.4f}")
    return start_epoch, best_valid_loss

def train_epoch(model, iterator, optimizer, criterion, clip, teacher_forcing_ratio):
    model.train()
    epoch_loss = 0.0

    for src, trg in iterator:
        src, trg = src.to(DEVICE), trg.to(DEVICE)
        optimizer.zero_grad()


        with torch.amp.autocast(device_type=DEVICE.type, enabled=(DEVICE.type == "cuda")):
            output = model(src, trg, teacher_forcing_ratio)
            output_dim = output.size(-1)
            output = output[:, 1:].reshape(-1, output_dim)
            trg_flat = trg[:, 1:].reshape(-1)
            loss = criterion(output, trg_flat)

        SCALER.scale(loss).backward()
        SCALER.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        SCALER.step(optimizer)
        SCALER.update()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0.0

    with torch.no_grad():
        for src, trg in iterator:
            src, trg = src.to(DEVICE), trg.to(DEVICE)

            with torch.amp.autocast(device_type=DEVICE.type, enabled=(DEVICE.type == "cuda")):
                output = model(src, trg, 0.0)  # eval doesn't use teacher forcing
                output_dim = output.size(-1)
                output = output[:, 1:].reshape(-1, output_dim)
                trg_flat = trg[:, 1:].reshape(-1)
                loss = criterion(output, trg_flat)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)

# ==============================================================================
# 5. Training Main Loop
# ==============================================================================

def train_model():
    print("\n--- Start Model Training (Attention + GloVe) ---")
    start_epoch, best_valid_loss = load_checkpoint(model, optimizer, MODEL_CHECKPOINT_PATH)

    for epoch in range(start_epoch, N_EPOCHS):
        start_time = time.time()
        train_loss = train_epoch(model, train_loader, optimizer, criterion, CLIP, TEACHER_FORCING_RATIO)

        valid_loss = evaluate(model, valid_loader, criterion)

        end_time = time.time()
        epoch_mins = int((end_time - start_time) / 60)
        epoch_secs = int((end_time - start_time) % 60)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            save_checkpoint(model, optimizer, epoch, best_valid_loss, MODEL_CHECKPOINT_PATH)
            print("\t==> Model performance improved, saved best checkpoint <==")

        print(f"Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s")
        print(f"\tTrain Loss: {train_loss:.4f} | Train PPL: {math.exp(train_loss):.2f}")
        print(f"\tValid Loss: {valid_loss:.4f} | Valid PPL: {math.exp(valid_loss):.2f}")

# ==============================================================================
# 6. Inference (Generate Response)
# ==============================================================================

def generate_response(sentence, model, w2i, i2w, max_len=MAX_LEN):
    model.eval()

    tokens = tokenize(sentence)
    indexed = [w2i[SOS_TOKEN]] + [w2i.get(t, w2i[UNK_TOKEN]) for t in tokens] + [w2i[EOS_TOKEN]]
    if len(indexed) < max_len:
        indexed += [w2i[PAD_TOKEN]] * (max_len - len(indexed))
    else:
        indexed = indexed[:max_len]

    src_tensor = torch.tensor(indexed, dtype=torch.long).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(src_tensor)

    trg_indexes = [w2i[SOS_TOKEN]]
    input_token = torch.tensor([w2i[SOS_TOKEN]], dtype=torch.long).to(DEVICE)

    for _ in range(max_len):
        with torch.no_grad():
            output, hidden, cell = model.decoder(input_token, hidden, cell, encoder_outputs)
        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token)

        if pred_token == w2i[EOS_TOKEN]:
            break

        input_token = torch.tensor([pred_token], dtype=torch.long).to(DEVICE)

    trg_tokens = [i2w[idx] for idx in trg_indexes if idx in i2w]

    if EOS_TOKEN in trg_tokens:
        eos_pos = trg_tokens.index(EOS_TOKEN)
        trg_tokens = trg_tokens[1:eos_pos]
    else:
        trg_tokens = trg_tokens[1:]

    return " ".join(trg_tokens)

# ==============================================================================
# 7. Main Entry Point
# ==============================================================================

if __name__ == "__main__":
    # train_model()

    if os.path.exists(MODEL_CHECKPOINT_PATH):
        print("\n--- Model Inference ---")
        # Add weights_only=False to allow loading old weights with Numpy data or complex dicts
        checkpoint = torch.load(MODEL_CHECKPOINT_PATH, map_location=DEVICE, weights_only=False)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)

        s1 = "I just lost my pet dog yesterday, and I feel so empty."

        r1 = generate_response(s1, model, word_to_idx, idx_to_word)

        print(f"Input: {s1}")
        print(f"Model Response: {r1}")

    else:
        print("\n--- Warning ---")
        print(f"Model checkpoint file '{MODEL_CHECKPOINT_PATH}' not found.")

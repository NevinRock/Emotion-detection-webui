import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import CountVectorizer
import time
import math
import random
import os

# ==============================================================================
# 0. Configuration & Initialization
# ==============================================================================

# Model and Data Config
MAX_LEN = 50        # Max sentence length
BATCH_SIZE = 256
EMBEDDING_DIM = 256
HIDDEN_DIM = 512
NUM_LAYERS = 2
DROPOUT = 0.7
N_EPOCHS = 10       # Training epochs
CLIP = 1.0          # Gradient clipping threshold
TEACHER_FORCING_RATIO = 0.5 # Teacher Forcing Ratio
SAMPLE_FRACTION = 1 # Dataset downsizing fraction (1.0 = use all data; 0.1 = use 10%)

# Device Settings
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device used: {DEVICE}")

# Special Tokens
PAD_TOKEN = '<pad>' # Padding token
SOS_TOKEN = '<sos>' # Start of sequence token
EOS_TOKEN = '<eos>' # End of sequence token
UNK_TOKEN = '<unk>' # Unknown word token
SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
MODEL_CHECKPOINT_PATH = 'best_empathetic_lstm_model.pt' # Checkpoint file path

# ==============================================================================
# 1. Data Loading & Vocab Construction
# ==============================================================================

def tokenize(text):
    """Convert text to lowercase word list"""
    return text.lower().split()

# Load Data
try:
    train_df = pd.read_csv(r"C:\Users\Harry\Desktop\DeepLearning_RAI\Dialogue_CW\empathetic_dialog_datasets\processed\train_processed.csv")
    valid_df = pd.read_csv(r"C:\Users\Harry\Desktop\DeepLearning_RAI\Dialogue_CW\empathetic_dialog_datasets\processed\valid_processed.csv")
except FileNotFoundError:
    print("Error: train_processed.csv or valid_processed.csv not found. Please ensure filename and path are correct.")
    exit()

# === Core fix for KeyError: Rename Columns ===
# Assuming columns from data cleaning are 'input_utterance' and 'output_utterance'
# We need to rename them to 'input_text' and 'target_text' as expected by training code
train_df = train_df.rename(columns={'input_utterance': 'input_text', 'output_utterance': 'target_text'})
valid_df = valid_df.rename(columns={'input_utterance': 'input_text', 'output_utterance': 'target_text'})

# Check rename success again (optional but recommended)
if 'input_text' not in train_df.columns:
    print("Critical Warning: Column rename failed! Please check if column names in train_processed.csv are 'input_utterance'.")
    exit()
# =======================================

print(f"Original Train Size: {len(train_df)}")

# Downsizing dataset (if SAMPLE_FRACTION < 1.0)
if SAMPLE_FRACTION < 1.0:
    train_df = train_df.sample(frac=SAMPLE_FRACTION, random_state=SEED).reset_index(drop=True)
    valid_df = valid_df.sample(frac=SAMPLE_FRACTION, random_state=SEED).reset_index(drop=True)
    print(f"Reduced train set to {SAMPLE_FRACTION*100:.0f}%, Current Size: {len(train_df)}")
    print(f"Reduced valid set to {SAMPLE_FRACTION*100:.0f}%, Current Size: {len(valid_df)}")


# Build Vocab
all_texts = pd.concat([train_df['input_text'], train_df['target_text']])

# Use CountVectorizer to build vocab
vectorizer = CountVectorizer(tokenizer=tokenize, max_features=20000)
vectorizer.fit(all_texts.astype(str)) # Ensure input is string

# Build Mapping
word_to_idx = {word: idx + len(SPECIAL_TOKENS) for word, idx in vectorizer.vocabulary_.items()}

# Add special tokens to mapping
word_to_idx[PAD_TOKEN] = 0
word_to_idx[SOS_TOKEN] = 1
word_to_idx[EOS_TOKEN] = 2
word_to_idx[UNK_TOKEN] = 3

VOCAB_SIZE = len(word_to_idx)
idx_to_word = {idx: word for word, idx in word_to_idx.items()}
TRG_PAD_IDX = word_to_idx[PAD_TOKEN]

print(f"Vocab Size (VOCAB_SIZE): {VOCAB_SIZE}")

# ==============================================================================
# 2. Dataset and DataLoader
# ==============================================================================

class DialogueDataset(Dataset):
    """Custom PyTorch Dataset class to convert text to sequence"""
    def __init__(self, df, w2i, max_len):
        # df already has 'input_text' and 'target_text' columns, KeyError fixed
        self.input_sequences = df['input_text'].tolist()
        self.target_sequences = df['target_text'].tolist()
        self.w2i = w2i
        self.max_len = max_len

    def __len__(self):
        return len(self.input_sequences)

    def __getitem__(self, idx):
        def text_to_sequence(text):
            if pd.isna(text): # Handle possible NaN values
                return torch.tensor([self.w2i[SOS_TOKEN], self.w2i[EOS_TOKEN]] + [self.w2i[PAD_TOKEN]] * (self.max_len - 2))

            tokens = tokenize(str(text))
            # Add SOS and EOS
            sequence = [self.w2i[SOS_TOKEN]] + [self.w2i.get(token, self.w2i[UNK_TOKEN]) for token in tokens] + [self.w2i[EOS_TOKEN]]
            # Truncate or Pad
            if len(sequence) > self.max_len:
                sequence = sequence[:self.max_len]
            elif len(sequence) < self.max_len:
                sequence = sequence + [self.w2i[PAD_TOKEN]] * (self.max_len - len(sequence))
            return torch.tensor(sequence)

        input_seq = text_to_sequence(self.input_sequences[idx])
        target_seq = text_to_sequence(self.target_sequences[idx])
        
        return input_seq, target_seq

train_dataset = DialogueDataset(train_df, word_to_idx, MAX_LEN)
valid_dataset = DialogueDataset(valid_df, word_to_idx, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)

# ==============================================================================
# 3. Seq2Seq LSTM Model Definition (Keep unchanged)
# ==============================================================================
# 

class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hid_dim, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        # input: [batch_size, 1]
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        # output: [batch_size, 1, hid_dim]
        prediction = self.fc_out(output.squeeze(1))
        # prediction: [batch_size, vocab_size]
        
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    """Sequence to Sequence (Seq2Seq) Model"""
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = trg.shape[0]
        trg_len = trg.shape[1]
        trg_vocab_size = self.decoder.vocab_size

        outputs = torch.zeros(batch_size, trg_len, trg_vocab_size).to(self.device)

        hidden, cell = self.encoder(src)

        input = trg[:, 0] # [batch_size]

        for t in range(1, trg_len):
            # input: [batch_size] -> Decoder needs [batch_size, 1]
            output, hidden, cell = self.decoder(input.unsqueeze(1), hidden, cell)

            outputs[:, t, :] = output

            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.argmax(1) 

            input = trg[:, t] if teacher_force else top1
            
        return outputs

# ==============================================================================
# 4. Model Instantiation, Checkpoint Resume & Train/Eval Functions
# ==============================================================================

# Instantiate Model
enc = Encoder(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
dec = Decoder(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, NUM_LAYERS, DROPOUT)
model = Seq2Seq(enc, dec, DEVICE).to(DEVICE)

# Loss Function: CrossEntropyLoss (Ignore PAD token loss)
criterion = nn.CrossEntropyLoss(ignore_index=TRG_PAD_IDX)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)


def save_checkpoint(model, optimizer, epoch, best_valid_loss, filename):
    """Function to save model, optimizer state and training progress"""
    state = {
        'epoch': epoch,
        'best_valid_loss': best_valid_loss,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'vocab_size': VOCAB_SIZE, 
        'max_len': MAX_LEN,
    }
    torch.save(state, filename)
    
def load_checkpoint(model, optimizer, filename):
    """Function to load model, optimizer state and training progress"""
    if os.path.exists(filename):
        print(f"==> Checkpoint found: {filename}, loading...")
        checkpoint = torch.load(filename, map_location=DEVICE)
        
        # Check config compatibility
        if checkpoint['vocab_size'] != VOCAB_SIZE or checkpoint['max_len'] != MAX_LEN:
            print("Warning: Checkpoint config mismatch, training from scratch.")
            return 0, float('inf')

        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1 # Start from next epoch
        best_valid_loss = checkpoint['best_valid_loss']
        print(f"==> Loaded successfully. Starting from Epoch {start_epoch}, Current Best Valid Loss: {best_valid_loss:.4f}")
        return start_epoch, best_valid_loss
    else:
        print("==> No checkpoint found, starting training from Epoch 1.")
        return 0, float('inf')


def train_epoch(model, iterator, optimizer, criterion, clip, teacher_forcing_ratio):
    model.train()
    epoch_loss = 0
    
    # Initialize GradScaler (for mixed precision training)
    scaler = torch.amp.GradScaler(enabled=(DEVICE.type == 'cuda'))
    
    for i, (src, trg) in enumerate(iterator):
        src, trg = src.to(DEVICE), trg.to(DEVICE)
        optimizer.zero_grad()
        
        # Enable Automatic Mixed Precision
        with torch.amp.autocast(device_type=DEVICE.type, enabled=(DEVICE.type == 'cuda')):
            output = model(src, trg, teacher_forcing_ratio)
            
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim) 
            trg = trg[:, 1:].reshape(-1)
            
            loss = criterion(output, trg)
        
        # Scale loss and backward pass
        scaler.scale(loss).backward()
        
        # Gradient Clipping (before scaler.step())
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        
        # Update Weights
        scaler.step(optimizer)
        scaler.update()
        
        epoch_loss += loss.item()

    return epoch_loss / len(iterator)

def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    
    with torch.no_grad():
        for i, (src, trg) in enumerate(iterator):
            src, trg = src.to(DEVICE), trg.to(DEVICE)
            
            with torch.amp.autocast(device_type=DEVICE.type, enabled=(DEVICE.type == 'cuda')):
                # Disable teacher forcing
                output = model(src, trg, 0) 
                
                output_dim = output.shape[-1]
                output = output[:, 1:].reshape(-1, output_dim)
                trg = trg[:, 1:].reshape(-1)
                
                loss = criterion(output, trg)
                epoch_loss += loss.item()

    return epoch_loss / len(iterator)

# ==============================================================================
# 5. Model Training (Supports Checkpoint Resume)
# ==============================================================================

def train_model():
    print("\n--- Start Training ---")
    
    # === [Checkpoint Resume] Load Checkpoint ===
    start_epoch, best_valid_loss = load_checkpoint(model, optimizer, MODEL_CHECKPOINT_PATH)
    # ==========================

    for epoch in range(start_epoch, N_EPOCHS):
        start_time = time.time()
        
        train_loss = train_epoch(model, train_loader, optimizer, criterion, CLIP, TEACHER_FORCING_RATIO)
        valid_loss = evaluate(model, valid_loader, criterion)
        
        end_time = time.time()
        epoch_mins = int((end_time - start_time) / 60)
        epoch_secs = int((end_time - start_time) % 60)
        
        # === [Checkpoint Resume] Save Best Checkpoint ===
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            save_checkpoint(model, optimizer, epoch, best_valid_loss, MODEL_CHECKPOINT_PATH) 
            print("\t==> Model performance improved, saved best checkpoint <==")
        # ================================
            
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.4f} | Train Perplexity: {math.exp(train_loss):.2f}')
        print(f'\tValid Loss: {valid_loss:.4f} | Valid Perplexity: {math.exp(valid_loss):.2f}')

# ==============================================================================
# 6. Inference (Dialogue Generation)
# ==============================================================================

def generate_response(sentence, model, w2i, i2w, max_len=MAX_LEN):
    model.eval()
    
    # 1. Preprocess input sentence
    tokens = tokenize(sentence)
    indexed = [w2i[SOS_TOKEN]] + [w2i.get(token, w2i[UNK_TOKEN]) for token in tokens] + [w2i[EOS_TOKEN]]
    src_tensor = torch.tensor(indexed[:max_len], dtype=torch.long).unsqueeze(0).to(DEVICE)
    
    # 2. Encode
    with torch.no_grad():
        hidden, cell = model.encoder(src_tensor)
        
        # Force hidden state memory layout to be contiguous
        hidden = hidden.contiguous()
        cell = cell.contiguous()

    # 3. Decode (Auto-regressive generation)
    trg_indexes = [w2i[SOS_TOKEN]]
    # Initial Input: [1] dim vector
    input_token = torch.tensor([w2i[SOS_TOKEN]], dtype=torch.long).to(DEVICE) 

    for _ in range(max_len):
        # input_token.unsqueeze(0) shape is [1, 1]
        input_for_decoder = input_token.unsqueeze(0) 
        
        with torch.no_grad():
            output, hidden, cell = model.decoder(input_for_decoder, hidden, cell)
        
        # Predict next token index
        pred_token = output.argmax(1).item()
        trg_indexes.append(pred_token)
        
        # Stop condition
        if pred_token == w2i[EOS_TOKEN]:
            break
            
        # Next time step input is current predicted token, kept as [1] dim tensor
        input_token = torch.tensor([pred_token], dtype=torch.long).to(DEVICE)
        
    # 4. Convert to text
    trg_tokens = [i2w[idx] for idx in trg_indexes]
    return ' '.join(trg_tokens[1:-1])

# ==============================================================================
# 7. Main Entry Point
# ==============================================================================
if __name__ == '__main__':
    
    # Start Training
    # train_model() 
    
    # After training, load best weights for inference
    if os.path.exists(MODEL_CHECKPOINT_PATH):
        print("\n--- Model Inference ---")
        try:
            # Load checkpoint and extract model state dict
            checkpoint = torch.load(MODEL_CHECKPOINT_PATH, map_location=DEVICE)
            model.load_state_dict(checkpoint['model_state_dict'])
            
            input_sentence = "I just lost my pet dog yesterday, and I feel so empty."
            
            response = generate_response(input_sentence, model, word_to_idx, idx_to_word)
            print(f"Input: {input_sentence}")
            print(f"Model Response: {response}")

            input_sentence_2 = "I went to the fireworks with my best friend. It was the best night ever."
            response_2 = generate_response(input_sentence_2, model, word_to_idx, idx_to_word)
            print(f"\nInput: {input_sentence_2}")
            print(f"Model Response: {response_2}")

        except RuntimeError as e:
            print(f"Error after loading model weights: {e}")
    else:
        print("\n--- Warning ---")
        print(f"Model checkpoint file '{MODEL_CHECKPOINT_PATH}' not found. Inference skipped or random initialization used.")
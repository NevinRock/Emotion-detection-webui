import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import random
import os

# ==============================================================================
# 1. 基础配置与工具函数
# ==============================================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 特殊标记
PAD_TOKEN = '<pad>'
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
UNK_TOKEN = '<unk>'

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
    """ 标准化分词工具 """
    text = str(text)
    text = EMOJI_PATTERN.sub(r"", text)
    text = text.lower()
    tokens = re.findall(r"[\w']+|[^\w\s]", text)
    return [t for t in tokens if t.strip()]

# ==============================================================================
# 2. 模型结构 (Encoder / Decoder / Seq2Seq)
# ==============================================================================

class Encoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        embedded = self.dropout(self.embedding(src))
        outputs, (hidden, cell) = self.rnn(embedded)
        return outputs, hidden, cell

class Decoder(nn.Module):
    def __init__(self, vocab_size, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.attn = nn.Linear(hid_dim * 2, hid_dim)
        self.v = nn.Linear(hid_dim, 1, bias=False)
        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc_out = nn.Linear(hid_dim * 2, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell, encoder_outputs):
        if input.dim() == 1:
            input = input.unsqueeze(1)
        
        embedded = self.dropout(self.embedding(input))
        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))
        
        # Attention
        hidden_top = hidden[-1]
        src_len = encoder_outputs.size(1)
        hidden_expanded = hidden_top.unsqueeze(1).repeat(1, src_len, 1)
        energy_input = torch.cat((hidden_expanded, encoder_outputs), dim=2)
        energy = torch.tanh(self.attn(energy_input))
        attention = F.softmax(self.v(energy).squeeze(2), dim=1)
        
        context = torch.bmm(attention.unsqueeze(1), encoder_outputs).squeeze(1)
        output_squeezed = output.squeeze(1)
        concat = torch.cat((output_squeezed, context), dim=1)
        prediction = self.fc_out(concat)
        
        return prediction, hidden, cell

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        # 仅用于训练，推断时不使用此 forward
        batch_size = trg.size(0)
        trg_len = trg.size(1)
        vocab_size = self.decoder.vocab_size
        outputs = torch.zeros(batch_size, trg_len, vocab_size, device=self.device)
        encoder_outputs, hidden, cell = self.encoder(src)
        input_token = trg[:, 0]
        for t in range(1, trg_len):
            output, hidden, cell = self.decoder(input_token, hidden, cell, encoder_outputs)
            outputs[:, t, :] = output
            top1 = output.argmax(1)
            input_token = trg[:, t] if random.random() < teacher_forcing_ratio else top1
        return outputs

# ==============================================================================
# 3. 封装好的处理器类 (Handler)
# ==============================================================================

class LocalLSTMHandler:
    def __init__(self, model_path):
        """
        初始化 Handler：加载 Checkpoint，恢复词表和模型结构
        """
        self.device = DEVICE
        self.is_loaded = False
        
        if not os.path.exists(model_path):
            print(f"❌ Error: Model path not found: {model_path}")
            return

        print(f"🔄 Loading LSTM model from {model_path}...")
        try:
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # 1. 恢复配置
            self.model_config = checkpoint.get('config', {})
            # 如果是旧版 checkpoint 没有 config，需要手动设置默认值 (这里的默认值根据你之前的代码设定)
            self.emb_dim = self.model_config.get('emb_dim', 100)
            self.hid_dim = self.model_config.get('hid_dim', 128)
            self.n_layers = self.model_config.get('n_layers', 2)
            self.dropout = self.model_config.get('dropout', 0.5)
            self.max_len = checkpoint.get('max_len', 50)

            # 2. 恢复词表 (这是最关键的一步)
            if 'word_to_idx' in checkpoint:
                self.word_to_idx = checkpoint['word_to_idx']
            else:
                raise ValueError("Checkpoint does not contain 'word_to_idx'. Please re-save model with vocab.")
                
            self.idx_to_word = {v: k for k, v in self.word_to_idx.items()}
            self.vocab_size = len(self.word_to_idx)
            
            # 3. 初始化模型结构
            enc = Encoder(self.vocab_size, self.emb_dim, self.hid_dim, self.n_layers, self.dropout)
            dec = Decoder(self.vocab_size, self.emb_dim, self.hid_dim, self.n_layers, self.dropout)
            self.model = Seq2Seq(enc, dec, self.device).to(self.device)
            
            # 4. 加载权重
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.model.eval()
            self.is_loaded = True
            print("✅ LSTM Model loaded successfully!")

        except Exception as e:
            print(f"❌ Failed to load model: {e}")
            self.is_loaded = False

    def chat(self, user_text, emotion_label=None, history_context=None):
        """
        对外接口：接收文本，返回模型生成的回复
        """
        if not self.is_loaded:
            return "[Error] LSTM Model is not loaded properly."

        # 可以选择把 emotion_label 加到输入前面，如果你训练时没有这样做，这里就不要加
        # input_text = f"{emotion_label} {user_text}" if emotion_label else user_text
        input_text = user_text 

        return self._generate_sequence(input_text)

    def _generate_sequence(self, sentence):
        """ 内部推断逻辑 """
        tokens = tokenize(sentence)
        
        # 数值化
        w2i = self.word_to_idx
        sos_idx = w2i.get(SOS_TOKEN, 1)
        eos_idx = w2i.get(EOS_TOKEN, 2)
        unk_idx = w2i.get(UNK_TOKEN, 3)
        pad_idx = w2i.get(PAD_TOKEN, 0)

        indexed = [sos_idx] + [w2i.get(t, unk_idx) for t in tokens] + [eos_idx]
        
        # Padding / Truncating
        if len(indexed) < self.max_len:
            indexed += [pad_idx] * (self.max_len - len(indexed))
        else:
            indexed = indexed[:self.max_len]

        src_tensor = torch.tensor(indexed, dtype=torch.long).unsqueeze(0).to(self.device)

        with torch.no_grad():
            encoder_outputs, hidden, cell = self.model.encoder(src_tensor)

        trg_indexes = [sos_idx]
        input_token = torch.tensor([sos_idx], dtype=torch.long).to(self.device)

        # Greedy Decoding
        for _ in range(self.max_len):
            with torch.no_grad():
                output, hidden, cell = self.model.decoder(input_token, hidden, cell, encoder_outputs)
            
            pred_token = output.argmax(1).item()
            trg_indexes.append(pred_token)

            if pred_token == eos_idx:
                break

            input_token = torch.tensor([pred_token], dtype=torch.long).to(self.device)

        # ID 转 文字
        trg_tokens = []
        for idx in trg_indexes:
            if idx in self.idx_to_word:
                word = self.idx_to_word[idx]
                trg_tokens.append(word)
        
        # 去除特殊标记 SOS, EOS
        result = []
        for t in trg_tokens:
            if t == SOS_TOKEN: continue
            if t == EOS_TOKEN: break
            if t == PAD_TOKEN: continue
            result.append(t)

        return " ".join(result)
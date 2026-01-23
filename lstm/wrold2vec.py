import numpy as np
import torch
import torch.nn as nn
import pandas as pd
import os
from sklearn.feature_extraction.text import CountVectorizer
import re

# ==============================================================================
# 0. Config & File Paths (Corrected for your latest environment)
# ==============================================================================

# Dimensions used in model config
EMBEDDING_DIM = 200  # ⬅️ Matches your loaded glove.6B.200d.txt file
MAX_FEATURES = 20000

# Special Tokens (Keep unchanged)
PAD_TOKEN = '<pad>'
SOS_TOKEN = '<sos>'
EOS_TOKEN = '<eos>'
UNK_TOKEN = '<unk>'
SPECIAL_TOKENS = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]

# CSV File Paths (Your local paths)
TRAIN_CSV_PATH = r"C:\Users\Harry\Desktop\DeepLearning_RAI\Dialogue_CW\empathetic_dialog_datasets\processed\train_processed.csv"
VALID_CSV_PATH = r"C:\Users\Harry\Desktop\DeepLearning_RAI\Dialogue_CW\empathetic_dialog_datasets\processed\valid_processed.csv"
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F600-\U0001F64F"  # Emoticons
    "\U0001F300-\U0001F5FF"  # Symbols & Pictographs
    "\U0001F680-\U0001F6FF"  # Transport & Map Symbols
    "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
    "\U00002600-\U000027B0"  # Miscellaneous Symbols
    "\\ufe0f"                # Variation Selector 16 (Handles composite Emoji)
    "]+",
    flags=re.UNICODE
)
# Pretrained Word Vector File Path (Your local path)
PRETRAINED_EMBEDDINGS_PATH = r'C:\Users\Harry\Desktop\DeepLearning_RAI\Dialogue_CW\glove.6B\glove.6B.200d.txt' 

# ==============================================================================
# 1. Vocab Creation (Consistent with training code)
# ==============================================================================
COMMON_PUNCTUATION_SET = r"[.,!?:;()\[\]{}'\"-]"
def tokenize(text):
    """
    Final Version Tokenizer:
    1. Remove Emoji
    2. Convert to Lowercase
    3. Separate punctuation using regex, while keeping standard abbreviations (e.g., "don't", "you're").
    """
    # 1. Remove Emoji
    text = EMOJI_PATTERN.sub(r'', text)
    
    # 2. Convert to Lowercase
    text = text.lower()
    
    # 3. Separate punctuation: Match words/abbreviations or single non-whitespace punctuation
    # [\w']+ : Matches letters, numbers, underscores or words with single quotes (keeps abbreviations)
    # [^\w\s] : Matches single punctuation, Emoji, etc. (non-word, non-whitespace characters)
    tokens = re.findall(r"[\w']+|[^\w\s]", text)
    
    # 4. Filter empty strings to ensure clean output
    return [t for t in tokens if t.strip()]

def create_vocab_from_csv(train_path, valid_path):
    """Read CSV file directly and create word_to_idx vocab"""
    try:
        train_df = pd.read_csv(train_path)
        valid_df = pd.read_csv(valid_path)
    except FileNotFoundError:
        print("Error: Train or Valid CSV file not found, please check path.")
        return None
    except Exception as e:
        print(f"Error: Failed to read CSV file: {e}")
        return None

    # Fix Key Error part: Rename columns to match training code
    train_df = train_df.rename(columns={'input_utterance': 'input_text', 'output_utterance': 'target_text'})
    valid_df = valid_df.rename(columns={'input_utterance': 'input_text', 'output_utterance': 'target_text'})

    # Merge all texts for vocab construction
    all_texts = pd.concat([train_df['input_text'], train_df['target_text']])

    # Use CountVectorizer to build vocab (Consistent with training code)
    vectorizer = CountVectorizer(tokenizer=tokenize, max_features=MAX_FEATURES, min_df=5)
    vectorizer.fit(all_texts.astype(str))

    # Build Mapping
    word_to_idx = {word: idx + len(SPECIAL_TOKENS) for word, idx in vectorizer.vocabulary_.items()}

    # Add Special Tokens
    word_to_idx[PAD_TOKEN] = 0
    word_to_idx[SOS_TOKEN] = 1
    word_to_idx[EOS_TOKEN] = 2
    word_to_idx[UNK_TOKEN] = 3
    
    return word_to_idx

# ==============================================================================
# 2. Pretrained Embedding Loading Function
# ==============================================================================

def load_pretrained_embeddings(path, w2i, emb_dim):
    """Load pretrained word vectors and generate weight matrix"""
    print(f"Target Vocab Size: {len(w2i)}, Target Embedding Dimension: {emb_dim}")
    
    # Initialize weight matrix: [VOCAB_SIZE, EMBEDDING_DIM]. Default all vectors to zero.
    weights_matrix = np.zeros((len(w2i), emb_dim), dtype='float32')
    words_found = 0
    
    if not os.path.exists(path):
        print(f"Error: Pretrained file {path} not found. Please check path.")
        return None, 0
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                # Ensure enough elements in line (word + emb_dim values)
                if len(parts) <= emb_dim: continue
                
                word = parts[0]
                
                if word in w2i:
                    idx = w2i[word]
                    try:
                        vector = np.asarray(parts[1:], dtype='float32')
                        
                        if vector.shape[0] == emb_dim:
                            weights_matrix[idx] = vector
                            words_found += 1
                            
                    except ValueError:
                        continue 

        print(f"Successfully matched {words_found} vocab vectors in pretrained file.")
        print(f"{len(w2i) - words_found} words not found (or dimension mismatch), using zero vectors.")

    except Exception as e:
        print(f"Unexpected error loading pretrained file: {e}")
        return None, 0

    return torch.from_numpy(weights_matrix), words_found

# ==============================================================================
# 3. Execute Test (RuntimeError Fixed)
# ==============================================================================

if __name__ == '__main__':
    print("\n--- Step 1: Create Vocab ---")
    word_to_idx = create_vocab_from_csv(TRAIN_CSV_PATH, VALID_CSV_PATH)
    
    if word_to_idx is None:
        exit()

    print(f"Vocab creation complete. Total Vocab Size (VOCAB_SIZE): {len(word_to_idx)}")
    # Sort vocab for easy viewing (sorted by index)
    sorted_vocab = sorted(word_to_idx.items(), key=lambda item: item[1])
    
    # Print last 200 items
    for word, idx in sorted_vocab[-200:]:
        print(f"Index {idx:4d}: {word}")

    print("\n--- Step 2: Load Pretrained Embeddings ---")
    pretrained_weights, count = load_pretrained_embeddings(
        PRETRAINED_EMBEDDINGS_PATH, 
        word_to_idx, 
        EMBEDDING_DIM
    )

    if pretrained_weights is not None and count > 0:
        print("\n--- Step 3: PyTorch Embedding Layer Verification ---")
        
        # Instantiate nn.Embedding layer and verify dimensions
        # freeze=False means these pretrained vectors will be updated during training
        embedding_layer = nn.Embedding.from_pretrained(pretrained_weights, freeze=False)
        print(f"Embedding layer initialized successfully. Tensor shape: {embedding_layer.weight.shape}")
        
        # Verify a common word ('the') is successfully loaded (its norm should be much greater than 0)
        test_word = 'the'
        if test_word in word_to_idx:
            test_idx = word_to_idx[test_word]
            
            # Core fix: Use .detach() to solve RuntimeError
            test_vector_norm = np.linalg.norm(embedding_layer.weight[test_idx].detach().cpu().numpy())
            
            print(f"Vector norm for test word '{test_word}': {test_vector_norm:.4f}")
            
            if test_vector_norm > 1e-4:
                print("Pretrained vectors loaded and aligned successfully! You can integrate this logic into your training code.")
            else:
                print("Warning: Test word norm is close to zero, loading might have failed or word not in pretrained file.")
    else:
        print("\nTest terminated. Pretrained file loading failed or no words matched.")
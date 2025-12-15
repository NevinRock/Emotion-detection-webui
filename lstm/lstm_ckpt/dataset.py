import pandas as pd
import os

# Define output file name
OUTPUT_FILENAME = "valid.csv"

# 1. Load local file
# Assume ori/valid.csv is in the working directory
print("Loading ori/valid.csv file...")
# Use on_bad_lines='skip' to ignore malformed rows
try:
    df = pd.read_csv("ori/valid.csv", on_bad_lines='skip')
except FileNotFoundError:
    print("Error: ori/valid.csv not found. Please ensure the file exists in the correct directory.")
    exit()

print(f"Successfully loaded {len(df)} rows of raw data.")

# List to store processed dialogue pairs
processed_data = []

# 2. Group by conversation ID
grouped = df.groupby('conv_id')

print("Reconstructing dialogue data...")

for conv_id, group in grouped:
    # 2. Sort by utterance index to ensure correct dialogue order
    dialogue = group.sort_values(by='utterance_idx')
    
    # Extract emotional context for the dialogue
    context = dialogue['context'].iloc[0]
    
    # 3. Iterate through dialogue turns to build input–target pairs
    for i in range(1, len(dialogue)):
        # Previous utterance as input, current utterance as target
        input_utterance = dialogue['utterance'].iloc[i - 1]
        target_response = dialogue['utterance'].iloc[i]
        
        # Final input format: [CONTEXT:emotion] + previous utterance
        final_input = f"[CONTEXT:{context}] {input_utterance}"
        
        processed_data.append({
            'input_text': final_input,
            'target_text': target_response
        })

# 4. Convert to DataFrame for saving
processed_df = pd.DataFrame(processed_data)

# 5. Save to local CSV file (core step)
print(f"Saving {len(processed_df)} processed rows to {OUTPUT_FILENAME}...")
processed_df.to_csv(OUTPUT_FILENAME, index=False, encoding='utf-8')

print("\n--- Task Completed ---")
print(f"Data successfully saved to: {os.path.abspath(OUTPUT_FILENAME)}")
print("\n--- Sample of Preprocessed Data ---")
print(processed_df.head().to_markdown(index=False))

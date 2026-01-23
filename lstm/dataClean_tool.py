import pandas as pd
import os

# Define output filename
OUTPUT_FILENAME = "test_processed.csv"

# 1. Load local file
print("Loading ori/valid.csv file...")
# Use on_bad_lines='skip' to skip malformed lines
try:
    df = pd.read_csv("empathetic_dialog_datasets/ori/test.csv", on_bad_lines='skip')
except FileNotFoundError:
    print("Error: ori/valid.csv not found. Please ensure file path is correct.")
    exit()

print(f"Successfully loaded {len(df)} lines of raw data.")

# List to store final training data
processed_data = []

# 2. Group by Conversation ID
grouped = df.groupby('conv_id')

print("Restructuring dialogue data...")

for conv_id, group in grouped:
    # 2. Sort by utterance index, ensure dialogue order is correct
    dialogue = group.sort_values(by='utterance_idx')
    
    # Extract dialogue 'emotion/context' label, assuming column name is context
    context = dialogue['context'].iloc[0]
    
    # 3. Iterate through dialogue turns, build input and target pairs
    for i in range(1, len(dialogue)):
        # Previous turn is Input, current turn is Target
        input_utterance = dialogue['utterance'].iloc[i - 1]
        target_response = dialogue['utterance'].iloc[i]
        
        # Construct final Input format: [CONTEXT:Emotion] + Previous utterance
        # final_input = f"[CONTEXT:{context}] {input_utterance}"

        final_input = f"{input_utterance}"
        
        # Here we save context as a separate column (emotion label)
        processed_data.append({
            'input': final_input,
            'output': target_response,
            'emotion': context,       # New: Emotion label column
        })

# 4. Convert to DataFrame for easy saving
processed_df = pd.DataFrame(processed_data)

# 5. Save to local CSV file
print(f"Saving processed {len(processed_df)} rows of data to {OUTPUT_FILENAME}...")
processed_df.to_csv(OUTPUT_FILENAME, index=False, encoding='utf-8')

print("\n--- Task Completed ---")
print(f"Data successfully saved to: {os.path.abspath(OUTPUT_FILENAME)}")
print("\n--- Processed Data Example ---")
print(processed_df.head().to_markdown(index=False))

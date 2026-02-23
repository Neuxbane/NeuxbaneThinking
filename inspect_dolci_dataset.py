from datasets import load_dataset
import json

# Download the Dolci-Think dataset
try:
    ds = load_dataset("Neelectric/Dolci-Think-SFT-7B_persona-if_Llama3_4096toks")
    print("Dataset loaded successfully.")
    
    # Check splits
    print(f"Splits: {list(ds.keys())}")
    
    # Print first few samples of the first split
    split_name = list(ds.keys())[0]
    for i in range(2):
        print(f"\n--- Sample {i} ---")
        print(json.dumps(ds[split_name][i], indent=2))

except Exception as e:
    print(f"Error: {e}")

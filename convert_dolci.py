from datasets import load_dataset
import json
import os

def main():
    dataset_name = "Neelectric/Dolci-Think-SFT-7B_persona-if_Llama3_4096toks"
    output_path = "/home/jupyter-240712834/research/NeuxbaneThinking/datasets/dolci-think.jsonl"
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"Loading dataset {dataset_name}...")
    try:
        ds = load_dataset(dataset_name, split="train")
        print(f"Dataset loaded with {len(ds)} items.")
    except Exception as e:
        print(f"Failed to load dataset: {e}")
        return

    print(f"Converting examples to {output_path}...")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for idx, item in enumerate(ds):
            source_messages = item.get('messages', [])
            conversations = []
            
            for msg in source_messages:
                role = msg.get('role', '')
                content = msg.get('content', '')
                # The <think> block is already in the content for this dataset.
                
                conversations.append({
                    "role": role,
                    "content": content
                })
            
            output_item = {
                "conversations": conversations,
                "tools": []
            }
            
            f.write(json.dumps(output_item, ensure_ascii=False) + '\n')
            
            if (idx + 1) % 50000 == 0:
                print(f"Processed {idx + 1} examples...")
                
    print(f"Successfully saved to {output_path}")

if __name__ == "__main__":
    main()

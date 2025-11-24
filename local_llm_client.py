import torch
import torch.nn as nn
# Import the model and the BLOCK_SIZE constant
from mini_transformer import MiniTransformer, BLOCK_SIZE 

# Path to the saved model file
MODEL_PATH = "mini_transformer.pth"

def load_and_run_local_model():
    # Load the saved state dictionary and metadata
    try:
        checkpoint = torch.load(MODEL_PATH)
    except FileNotFoundError:
        print(f"Error: Model file '{MODEL_PATH}' not found.")
        print("Please ensure you have run 'python3 mini_transformer.py' first.")
        return
        
    # FIX: Calculate the exact VOCAB_SIZE from the loaded data (stoi mapping)
    stoi = checkpoint['stoi']
    itos = checkpoint['itos']
    actual_vocab_size = len(stoi) 
    
    # FIX: Re-initialize the model structure using the actual size
    model = MiniTransformer(vocab_size=actual_vocab_size)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval() # Set model to evaluation mode
    
    # Define encoder/decoder functions using loaded mappings
    encode = lambda s: torch.tensor([stoi[c] for c in s], dtype=torch.long)
    decode = lambda l: ''.join([itos[i] for i in l])
    
    print(f"Local Mini-Transformer loaded and ready (VOCAB_SIZE: {actual_vocab_size}).")
    
    while True:
        user_input = input("\nEnter starting text (or 'exit'): ")
        if user_input.lower() == 'exit':
            break
            
        # Basic input validation and preparation
        try:
            context = encode(user_input).unsqueeze(0) # (1, T)
        except KeyError as e:
            print(f"Error: Character {e} not in model's vocabulary. Use characters from the training text (e.g., 'hello', 't', 's').")
            continue
        
        if context.size(1) == 0:
             print("Please enter some text to start generation.")
             continue
        
        if context.size(1) > BLOCK_SIZE:
             # Truncate if the context is too long
             context = context[:, -BLOCK_SIZE:] 
             
        # Generate the sequence
        generated_indices = model.generate(context, max_new_tokens=20)[0].tolist()
        
        # Decode the result
        generated_text = decode(generated_indices)
        
        print("\n--- Model Output ---")
        print(generated_text)
        print("--------------------")

if __name__ == '__main__':
    load_and_run_local_model()
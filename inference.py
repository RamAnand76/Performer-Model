# inference.py
import torch
from transformers import GPT2Tokenizer
from model.performer import PerformerModel
import os

# Hyperparameters (matching the training setup)
vocab_size = 50257  # GPT-2 tokenizer vocab size
embed_size = 256
num_heads = 8
hidden_size = 512
num_layers = 4
max_seq_len = 50
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load the tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Add padding token if it doesn't exist
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Function to load the trained model
def load_model(model_path, vocab_size, embed_size, num_heads, hidden_size, num_layers, max_seq_len):
    model = PerformerModel(vocab_size, embed_size, num_heads, hidden_size, num_layers, max_seq_len)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    return model

# Load the trained model
model_path = "./Performer/model.pth"
model = load_model(model_path, vocab_size, embed_size, num_heads, hidden_size, num_layers, max_seq_len)

# Generate mask for the target sequence (causal mask)
def generate_causal_mask(size):
    mask = torch.triu(torch.ones(size, size), diagonal=1).to(device)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask

# Function to generate a response given an input sequence
def generate_response(model, tokenizer, input_text, max_seq_len, max_response_len=50):
    # Tokenize the input
    input_ids = tokenizer(input_text, return_tensors="pt", max_length=max_seq_len, truncation=True, padding='max_length')['input_ids'].to(device)
    
    # Start the response generation with the [BOS] token (or equivalent)
    response_ids = input_ids[:, :1]  # Start with the first token of the input or you can use tokenizer.bos_token

    # Generate tokens iteratively
    for _ in range(max_response_len):
        # Create causal mask
        tgt_mask = generate_causal_mask(response_ids.size(1))
        
        # Pass the input and generated tokens through the model
        with torch.no_grad():
            output = model(input_ids, response_ids, tgt_mask=tgt_mask)
        
        # Get the logits for the last token and pick the token with the highest probability
        next_token_logits = output[:, -1, :]
        next_token_id = torch.argmax(next_token_logits, dim=-1).unsqueeze(0)
        
        # Concatenate the new token to the response_ids
        response_ids = torch.cat([response_ids, next_token_id], dim=1)
        
        # Stop if EOS token is generated
        if next_token_id.item() == tokenizer.eos_token_id:
            break

    # Decode the generated response
    response_text = tokenizer.decode(response_ids.squeeze(), skip_special_tokens=True)
    return response_text

# Example usage
if __name__ == "__main__":
    input_text = "Hello,"
    response = generate_response(model, tokenizer, input_text, max_seq_len)
    print(f"Input: {input_text}")
    print(f"Response: {response}")

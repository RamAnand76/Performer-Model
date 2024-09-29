# main.py
import torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer
from data.dataset import ConversationDataset
from model.performer import PerformerModel
from training.train import train_model
from data.process_cornell_data import load_and_preprocess_cornell
import os

# Hyperparameters
vocab_size = 50257  # GPT-2 tokenizer vocab size
embed_size = 256
num_heads = 8
hidden_size = 512
num_layers = 4
max_seq_len = 50
batch_size = 32
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load tokenizer
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Add padding token if it does not exist
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Load and preprocess the Cornell Movie Dialogs dataset
data_dir = "cornell_movie_dialogs_corpus"
conversations = load_and_preprocess_cornell(data_dir)

# Initialize dataset and dataloader
train_dataset = ConversationDataset(conversations, tokenizer, max_len=max_seq_len)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize model, optimizer, and loss function
model = PerformerModel(vocab_size, embed_size, num_heads, hidden_size, num_layers, max_seq_len).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = torch.nn.CrossEntropyLoss()

# Train the model
train_model(model, train_dataloader, optimizer, criterion, device, num_epochs=2)

os.makedirs("./Performer", exist_ok=True)

# Save the model's state_dict
torch.save(model.state_dict(), "./Performer/model.pth")
model.save_pretrained("./Performer")


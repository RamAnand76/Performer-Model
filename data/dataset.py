# data/dataset.py
import torch
from torch.utils.data import Dataset
from transformers import GPT2Tokenizer

class ConversationDataset(Dataset):
    def __init__(self, conversations, tokenizer, max_len):
        self.conversations = conversations
        self.tokenizer = tokenizer
        self.max_len = max_len

        # Ensure the tokenizer has a padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def __len__(self):
        return len(self.conversations)
    
    def __getitem__(self, idx):
        src, tgt = self.conversations[idx]
        
        src_tokens = self.tokenizer(src, padding='max_length', max_length=self.max_len, truncation=True, return_tensors="pt")['input_ids'].squeeze(0)
        tgt_tokens = self.tokenizer(tgt, padding='max_length', max_length=self.max_len, truncation=True, return_tensors="pt")['input_ids'].squeeze(0)
        
        return src_tokens, tgt_tokens

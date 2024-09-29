import os
import json
import torch
import torch.nn as nn
from performer_pytorch import Performer

class PerformerModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, hidden_size, num_layers, max_seq_len, dropout=0.1):
        super(PerformerModel, self).__init__()
        
        # Embedding layer for input tokens
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # Positional encodings for sequence order
        self.positional_encoding = self._generate_positional_encoding(max_seq_len, embed_size)
        
        # Performer block - replacing the standard transformer block
        self.performer = Performer(
            dim=embed_size,  # embedding dimension
            depth=num_layers,  # number of layers (encoder layers)
            heads=num_heads,  # number of attention heads
            causal=True,  # if you're doing autoregressive tasks, set causal=True
            ff_mult=4,  # feedforward hidden layer size is usually larger (like 4x embedding size)
            dropout=dropout  # dropout for regularization
        )
        
        # Linear output layer projecting back to vocab size
        self.fc_out = nn.Linear(embed_size, vocab_size)
        
        # Layer normalization for stability
        self.layer_norm = nn.LayerNorm(embed_size)
        
        # Model config to be saved
        self.config = {
            "vocab_size": vocab_size,
            "embed_size": embed_size,
            "num_heads": num_heads,
            "hidden_size": hidden_size,
            "num_layers": num_layers,
            "max_seq_len": max_seq_len,
            "dropout": dropout
        }

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Embedding the input tokens
        src_embed = self.embedding(src) + self.positional_encoding[:src.size(1), :]
        tgt_embed = self.embedding(tgt) + self.positional_encoding[:tgt.size(1), :]

        # Mask handling
        if src_mask is not None:
            src_mask = self._create_padding_mask(src)
        if tgt_mask is not None:
            tgt_mask = self._create_padding_mask(tgt)
        
        # Apply Performer attention with the masks
        src_embed = self.layer_norm(src_embed)
        tgt_embed = self.layer_norm(tgt_embed)
        
        output = self.performer(src_embed, tgt_embed, mask=src_mask)
        
        # Pass through the final fully connected layer to map to vocab size
        output = self.fc_out(output)
        return output

    def _generate_positional_encoding(self, max_seq_len, embed_size):
        """Generates sinusoidal positional encodings."""
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_size, 2).float() * (-torch.log(torch.tensor(10000.0)) / embed_size))
        pe = torch.zeros(max_seq_len, embed_size)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        return pe

    def _create_padding_mask(self, seq):
        """Creates a padding mask for the input sequences."""
        return (seq == 0).unsqueeze(1).unsqueeze(2)  # Mask shape [batch_size, 1, 1, seq_len]
    
    def save_pretrained(self, save_directory):
        """
        Save the model weights and configuration in Hugging Face format.
        """
        if not os.path.exists(save_directory):
            os.makedirs(save_directory)
        
        # Save the model weights (state_dict)
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), model_path)

        # Save the configuration
        config_path = os.path.join(save_directory, "config.json")
        with open(config_path, "w") as f:
            json.dump(self.config, f)

    @classmethod
    def from_pretrained(cls, load_directory):
        """
        Load the model weights and configuration from Hugging Face format.
        """
        # Load the configuration
        config_path = os.path.join(load_directory, "config.json")
        with open(config_path, "r") as f:
            config = json.load(f)

        # Create a new instance of the model with the loaded config
        model = cls(**config)

        # Load the model weights (state_dict)
        model_path = os.path.join(load_directory, "pytorch_model.bin")
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

        return model

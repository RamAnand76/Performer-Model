# model/performer.py
import torch
import torch.nn as nn
from torch.nn import Transformer

class PerformerModel(nn.Module):
    def __init__(self, vocab_size, embed_size, num_heads, hidden_size, num_layers, max_seq_len):
        super(PerformerModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.transformer = Transformer(d_model=embed_size, nhead=num_heads, num_encoder_layers=num_layers,
                                       num_decoder_layers=num_layers, dim_feedforward=hidden_size, batch_first=True)
        self.fc_out = nn.Linear(embed_size, vocab_size)
        self.max_seq_len = max_seq_len

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src_embed = self.embedding(src)
        tgt_embed = self.embedding(tgt)

        # Ensure src_mask and tgt_mask have the right shape
        if src_mask is not None:
            src_mask = self._create_padding_mask(src.size(1), src_mask.device, src.size(0))
        if tgt_mask is not None:
            tgt_mask = self._create_padding_mask(tgt.size(1), tgt_mask.device, tgt.size(0))

        output = self.transformer(src_embed, tgt_embed, src_key_padding_mask=src_mask, tgt_key_padding_mask=tgt_mask)
        output = self.fc_out(output)
        return output

    def _create_padding_mask(self, seq_len, device, batch_size):
        # Create a mask of shape (batch_size, seq_len)
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
        return mask

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


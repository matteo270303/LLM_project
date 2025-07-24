import torch.nn as nn
from config import GPT_CONFIG_124M
from Layer import TransformerLayer
import torch

class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tok_emb = nn.Embedding(config["vocab_size"], config["embedding_size"])
        self.pos_emb = nn.Embedding(config["context_length"], config["embedding_size"])
        self.dropout = nn.Dropout(config["drop_rate"])

        self.transformerBlock = nn.Sequential(
            *[TransformerLayer(
                d_in=config["embedding_size"],
                d_out=config["embedding_size"],
                num_heads=config["num_heads"],
                context_length=config["context_length"],
                dropout=config["drop_rate"],
                qkv_bias=config["qkv_bias"]
            ) for _ in range(config["num_layers"])]
        )

        self.ln_f = nn.LayerNorm(config["embedding_size"])
        self.head = nn.Linear(config["embedding_size"], config["vocab_size"], bias=False)

    def forward(self, x):
        batch_size, seq_length = x.size()
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(torch.arange(seq_length, device=x.device))
        x = tok_emb + pos_emb.unsqueeze(0)
        x = self.dropout(x)
        x = self.transformerBlock(x)
        if isinstance(x, tuple):
            x = x[0]
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

import torch.nn as nn
from config import Config
from Layer import TransformerLayer
from Embedding import Embedding
import torch

class GPTModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.tok_emb = nn.Embedding(config.vocab_size, config.embed_size)
        self.pos_emb = nn.Embedding(config.max_position_embeddings, config.embed_size)
        self.dropout = nn.Dropout(config.dropout)

        self.transformerBlock = nn.Sequential(
            *[TransformerLayer(
                d_in=config.embed_size,
                d_out=config.embed_size,
                num_heads=config.num_heads,
                context_length=config.max_position_embeddings,
                dropout=config.dropout,
                qkv_bias=True
            ) for _ in range(config.num_layers)]
        )

        self.ln_f = nn.LayerNorm(config.embed_size)
        self.head = nn.Linear(config.embed_size, config.vocab_size, bias=False)
    def forward(self, x):
        batch_size, seq_length = x.size()
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(torch.arange(seq_length, device=x.device))
        x = tok_emb + pos_emb.unsqueeze(0)
        x = self.dropout(x)
        x = self.transformerBlock(x)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

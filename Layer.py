import torch
import torch.nn as nn
from Function import GELU
from torch.nn.functional import softmax

class LayerNormalization(nn.Module):
    def __init__(self, embedding_size):
        super().__init__()
        self.epsilon = 1e-5
        self.gamma = nn.Parameter(torch.ones(embedding_size))  # Parametro trainabile
        self.beta = nn.Parameter(torch.zeros(embedding_size))  # Parametro trainabile

    def forward(self, x):
        # Calcola la media e la varianza lungo l'ultima dimensione (embedding_size)
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)

        # Normalizza l'input
        x_normalized = (x - mean) / torch.sqrt(var + self.epsilon)

        # Applica i parametri di scala (gamma) e shift (beta)
        return self.gamma * x_normalized + self.beta
    
class FeedForward(nn.Module):
    def __init__(self, embedding_size, hidden_size):
        super().__init__()
        self.embedding_size = embedding_size  # Memorizza la dimensione dell'input
        self.layer = nn.Sequential(
            nn.Linear(embedding_size, hidden_size),
            GELU(),
            nn.Linear(hidden_size, embedding_size)
        )

    def forward(self, x):
        # Controlla che la dimensione dell'input corrisponda a embedding_size
        if x.shape[-1] != self.embedding_size:  # Usa l'ultima dimensione per il controllo
            raise ValueError(f"Input shape {x.shape} does not match expected shape (..., {self.embedding_size})")

        return self.layer(x)
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, num_heads, context_length, dropout=0.4, qkv_bias=False):
        super().__init__()
        #assert self.head_dim * num_heads == d_out, "d_out must be divisible by num_heads"

        self.num_heads = num_heads
        self.d_out = d_out
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(d_out, d_out, bias=qkv_bias)
        self.register_buffer('mask',
                             torch.tril(torch.ones(context_length, context_length),
                                        diagonal=1))

    def forward(self, x):
        b, num_token, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # Reshape for multi-head attention
        keys = keys.view(b, num_token, self.num_heads, self.head_dim).transpose(1, 2)
        queries = queries.view(b, num_token, self.num_heads, self.head_dim).transpose(1, 2)
        values = values.view(b, num_token, self.num_heads, self.head_dim).transpose(1, 2)

        # Compute attention scores
        att_scores = torch.matmul(queries, keys.transpose(2, 3))  # [batch_size, num_heads, seq_len, seq_len]

        # Apply the mask using torch.tril
        mask = torch.tril(torch.ones(num_token, num_token, device=att_scores.device))  # Lower triangular mask
        mask = mask.unsqueeze(0).unsqueeze(1)  # [1, 1, seq_len, seq_len]
        att_scores = att_scores.masked_fill(mask == 0, float('-inf'))  # Mask invalid positions

        # Debugging: Check the mask
        print("Mask applied to attention scores:", mask)

        # Apply softmax to normalize scores
        att_weights = softmax(att_scores / self.head_dim**0.5, dim=-1)
        att_weights = self.dropout(att_weights)

        # Compute context vectors
        context_vec = torch.matmul(att_weights, values).transpose(1, 2)  # [batch_size, seq_len, num_heads, head_dim]
        context_vec = context_vec.contiguous().view(b, num_token, self.d_out)  # [batch_size, seq_len, d_out]

        # Apply final linear projection
        output = self.out_proj(context_vec)

        # Debugging: Print intermediate values
        print("Attention scores (after masking):", att_scores)
        print("Attention weights (after softmax):", att_weights)

        return output, att_weights

class TransformerLayer(nn.Module):
    def __init__(self, d_in, d_out, num_heads, context_length, dropout=0.1, qkv_bias=False):
        super().__init__()
        self.attention = MultiHeadAttention(d_in, d_out, num_heads, context_length, dropout, qkv_bias)
        self.norm1 = LayerNormalization(d_in)
        self.norm2 = LayerNormalization(d_in)
        self.ffn = FeedForward(d_in, d_out)

    def forward(self, x):
        att_output, att_weights = self.attention(x)
        x = self.norm1(x + att_output)  # Residual connection
        ffn_output = self.ffn(x)
        x = self.norm2(x + ffn_output)  # Residual connection
        return x, att_weights
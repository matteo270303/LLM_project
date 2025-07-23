import numpy as np 
from torch.nn.functional import softmax
import torch
import torch.nn as nn

class SelfAttention: 
    def __init__(self, input_dim, output_dim):
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weights = self.initialize_weights()

    def initialize_weights(self):
        # Initialize weights for the attention mechanism
        Q = torch.nn.Parameter(torch.randn(self.input_dim, self.output_dim))
        K = torch.nn.Parameter(torch.randn(self.input_dim, self.output_dim))
        V = torch.nn.Parameter(torch.randn(self.input_dim, self.output_dim))
        return {'query': Q, 'key': K, 'value': V}

    def compute_attention(self, inputs):
        # Compute queries, keys, and values
        queries = torch.matmul(inputs, self.weights['query'])
        keys = torch.matmul(inputs, self.weights['key'])
        values = torch.matmul(inputs, self.weights['value'])

        print("Queries shape:", queries.shape)
        print("Keys shape:", keys.shape)
        print("Values shape:", values.shape)

        # Compute attention scores
        scores = torch.matmul(queries, keys.transpose(1, 2)) / torch.sqrt(torch.tensor(self.output_dim, dtype=torch.float32))
        print("Scores shape:", scores.shape)

        # Create and apply attention mask
        batch_size, seq_len, _ = scores.shape
        mask = torch.tril(torch.ones(seq_len, seq_len)).to(scores.device)  # Lower triangular mask
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1)  # Expand mask to match batch size
        scores = scores.masked_fill(mask == 0, float('-inf'))  # Mask out positions with -inf

        # Apply softmax to normalize scores
        attention_weights = torch.nn.functional.softmax(scores, dim=-1)
        print("Attention weights shape:", attention_weights.shape)
        print("Attention weights (normalized):", attention_weights)

        # Compute the output
        output = torch.matmul(attention_weights, values)
        return output, attention_weights

    def attention_mask(self, scores):
        # Create an attention mask to prevent attending to certain positions
        context_length = scores.shape[-1]
        mask = torch.tril(torch.ones(context_length, context_length))
        print("Attention mask shape:", mask.shape)
        print("Attention mask:", mask)
        return mask

class CausalAttention(nn.Module): 
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.dropout = nn.Dropout(dropout)
        self.qkv_bias = qkv_bias
        self.W_wq = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_wk = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_wv = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.register_buffer('mask', 
                             torch.tril(torch.ones(context_length, context_length),
                                        diagonal=1))

    def forward(self, x):
        b, num_token, d_in = x.shape
        q = self.W_wq(x)
        k = self.W_wk(x)
        v = self.W_wv(x)

        att_scores = torch.matmul(q, k.transpose(1,2)) 
        att_scores.masked_fill_(self.mask.bool()[:num_token, :num_token], -torch.inf)
        att_weights = softmax(att_scores/k.shape[-1]**0.5, dim=-1)
        att_weights = self.dropout(att_weights)

        att_output = torch.matmul(att_weights, v)
        return att_output, att_weights
    
class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, num_heads, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList([
            CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
            for _ in range(num_heads)
        ])

    def forward(self, x):
        # Estrai l'output dell'attenzione (att_output) e i pesi (att_weights) da ciascuna testa
        att_outputs = []
        att_weights = []
        for head in self.heads:
            output, weights = head(x)  # Ottieni sia l'output che i pesi
            att_outputs.append(output)
            att_weights.append(weights)

        # Concatena gli output delle teste lungo l'ultima dimensione
        concatenated_output = torch.cat(att_outputs, dim=-1)  # [batch_size, seq_len, num_heads * d_out]

        # Concatena i pesi di attenzione lungo la dimensione batch
        concatenated_weights = torch.stack(att_weights, dim=1)  # [batch_size, num_heads, seq_len, seq_len]

        return concatenated_output, concatenated_weights
    

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
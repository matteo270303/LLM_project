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
        Q = torch.nn.Parameter(torch.randn(self.input_dim, self.output_dim))
        K = torch.nn.Parameter(torch.randn(self.input_dim, self.output_dim))
        V = torch.nn.Parameter(torch.randn(self.input_dim, self.output_dim))
        return {'query': Q, 'key': K, 'value': V}

    def compute_attention(self, inputs):
        queries = torch.matmul(inputs, self.weights['query'])
        keys = torch.matmul(inputs, self.weights['key'])
        values = torch.matmul(inputs, self.weights['value'])

        print("Queries shape:", queries.shape)
        print("Keys shape:", keys.shape)
        print("Values shape:", values.shape)

        scores = torch.matmul(queries, keys.transpose(1, 2)) / torch.sqrt(torch.tensor(self.output_dim, dtype=torch.float32))
        print("Scores shape:", scores.shape)

        batch_size, seq_len, _ = scores.shape
        mask = torch.tril(torch.ones(seq_len, seq_len)).to(scores.device)  #maschera triangolare inferiore 
        mask = mask.unsqueeze(0).expand(batch_size, -1, -1) 
        scores = scores.masked_fill(mask == 0, float('-inf')) # out position con -inf

        # normalizzazione con la softmax
        attention_weights = torch.nn.functional.softmax(scores, dim=-1)
        print("Attention weights shape:", attention_weights.shape)
        print("Attention weights (normalized):", attention_weights)

        output = torch.matmul(attention_weights, values)
        return output, attention_weights

    def attention_mask(self, scores):
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
    


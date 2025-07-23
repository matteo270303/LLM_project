import torch.nn as nn
import torch

class Embedding: 
    def create_embedding(self, vocab_size, embed_size):
        self.embeddings = nn.Embedding(vocab_size, embed_size)
        print(type(self.embeddings))
        return self.embeddings
    
    def create_token_embedding_layer_positional(self, vocab_size, embed_size, max_length):
        self.positional_embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_embeddings = self.positional_embedding(torch.arange(max_length))
        print(self.pos_embeddings.shape)
        print(type(self.pos_embeddings))
        return self.pos_embeddings
    
    def create_input_embedding(self, embedding_layer, pos_embedding, input_ids):
        # Applica l'embedding agli input_ids per ottenere un tensore
        token_emb = embedding_layer(input_ids)  # tensor [batch, seq_len, embed_dim] o [seq_len, embed_dim]
        print("Token embedding type:", type(token_emb))
        print("Token embedding shape:", token_emb.shape)
        # Adatta la positional embedding alla shape di token_emb
        pos_emb = pos_embedding[:token_emb.shape[1], :].unsqueeze(0)  # [1, seq_len, embed_dim]
        print("Positional embedding type:", type(pos_emb))
        print("Positional embedding shape:", pos_emb.shape)
        # Somma i tensori
        input_embedding = token_emb + pos_emb
        print("Input embedding type:", type(input_embedding))
        print("Input embedding shape:", input_embedding.shape)
        return input_embedding
import torch.nn as nn
import torch

class Embedding: 
    def create_embedding(self, vocab_size, embed_size):
        self.embeddings = nn.Embedding(vocab_size, embed_size)
        return self.embeddings
    
    def create_token_embedding_layer_positional(self, vocab_size, embed_size, max_length=512):
        self.positional_embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_embeddings = self.positional_embedding(torch.arange(max_length))
        print (self.pos_embeddings.shape)
        return self.pos_embeddings
    
    def create_input_embedding(self):
        input_embedding = self.embeddings + self.pos_embeddings
        print(input_embedding.shape)
        return input_embedding
import os
from Function import generate_text
import torch
from Tokenizer import Tokenizer
from DataLoader import CustomDataset
from Embedding import Embedding
import torch.nn as nn
from Model import GPTModel
from config import GPT_CONFIG_124M

verdict_path = os.path.join("data", "verdict.txt")

with open (verdict_path, "r", encoding="UTF-8") as file:
    text = file.read()

tokenizer = Tokenizer(text)

# Effettuo l'encoding in interi
integer = tokenizer.encode()

print("Numero di token:", len(integer))

# Creo il DataLoader
data_loader = CustomDataset.create_dataloader(
    txt=text,
    batch_size=32,
    shuffle=False,
    max_length=512,
    stride=4
)

data_iter = iter(data_loader)
batch = next(data_iter)
input_ids = batch['input_ids']
target_ids = batch['target_ids']
print("Token id input:", input_ids)
print("Input shape:", input_ids.shape)

# Stampo la lunghezza del vocabolario
vocab_length = tokenizer.vocab_len()
print("Lunghezza del vocabolario:", vocab_length)

# creazione del livello di embedding in pytorch
embedding = Embedding()
embedding_layer = embedding.create_embedding(vocab_length, 768)
embedding_token_layer_positional = embedding.create_token_embedding_layer_positional(vocab_length, 768, max_length=512)
input_embedding = embedding.create_input_embedding(embedding_layer, embedding_token_layer_positional, input_ids)

'''# Processo di Multi-Head Attention
multi_head_attention = MultiHeadAttention(
    d_in=768,  # Dimensione dell'input
    d_out=768,  # Dimensione dell'output totale
    num_heads=1,  # Numero di teste
    context_length=input_embedding.shape[1],  # Lunghezza della sequenza
    dropout=0.1, 
    qkv_bias=True  # Bias per Q, K, V
)

# Passa l'input embedding attraverso il livello di Multi-Head Attention
attention_output, attention_weights = multi_head_attention(input_embedding)

# Stampa i risultati
print("Output della Multi-Head Attention:", attention_output)
print("Shape dell'output della Multi-Head Attention:", attention_output.shape)
print("Pesi di attenzione della Multi-Head Attention:", attention_weights)
print("Shape dei pesi di attenzione:", attention_weights.shape)'''

'''transformer_layer = TransformerLayer(
    d_in=768,  # Dimensione dell'input
    d_out=768,  # Dimensione dell'output
    num_heads=1,  # Numero di teste
    context_length=input_embedding.shape[1],  # Lunghezza della sequenza
    dropout=0.1,  # Dropout
    qkv_bias=True  # Bias per Q, K, V
)

# Passa l'input embedding attraverso il Transformer Layer
output, attention_weights = transformer_layer(input_embedding)

# Stampa i risultati
print("Output del Transformer Layer:", output)
print("Shape dell'output del Transformer Layer:", output.shape)
print("Pesi di attenzione del Transformer Layer:", attention_weights)
print("Shape dei pesi di attenzione:", attention_weights.shape)'''

torch.manual_seed(123)
model = GPTModel(GPT_CONFIG_124M)
output = model(input_ids)
print("Output del modello GPT:", output)
print("Shape dell'output del modello GPT:", output.shape)
print(output)

num_params = model.num_parameters()
print("Numero di parametri del modello GPT:", num_params)

#Esempio di generazione del testo
input_text = "Hello, I am"
encoded_input = tokenizer.encode(input_text)
input_tensor = torch.tensor(encoded_input).unsqueeze(0)  

print("Encoding testo di input:", encoded_input)
print("Input tensor per la generazione:", input_tensor)

model.eval()
out = generate_text(model = model, idx = input_tensor, max_new_tokens=10, context_length=GPT_CONFIG_124M["context_length"])

print("Testo generato:", out)
print("Testo generato decodificato:", tokenizer.decode(out.squeeze(0).tolist()))
import os
import Tokenizer
import DataLoader
import Embedding
import torch.nn as nn

verdict_path = os.path.join("data", "verdict.txt")

with open (verdict_path, "r", encoding="UTF-8") as file:
    text = file.read()


tokenizer = Tokenizer.Tokenizer(text)

# Effettuo l'encoding in interi
integer = tokenizer.encode()

print(integer)

# Stampo la decodifica della lista degli interi
print(tokenizer.decode(integer))
print("Numero di token:", len(integer))

# Creo il DataLoader
data_loader = DataLoader.CustomDataset.create_dataloader(
    txt=text,
    batch_size=8,
    shuffle=False,
    max_length=4,
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
embedding = Embedding.Embedding()
embedding_layer = embedding.create_embedding(vocab_length, 256)
embedding_token_layer_positional = embedding.create_token_embedding_layer_positional(vocab_length, 256, max_length=4)

# Applica l'embedding agli input_ids per ottenere un torch tensor
embedded_tokens = embedding_layer(input_ids)
print("Embedded tokens shape:", embedded_tokens.shape)
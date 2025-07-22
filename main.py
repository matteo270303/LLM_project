import os
import Tokenizer
import DataLoader

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

data_loader = DataLoader.CustomDataset.create_dataloader(
    txt=text,
    batch_size=8,
    shuffle=False,
    max_length=4,
    stride=4
)

data_iter = iter(data_loader)
batch = next(data_iter)
print("Input IDs:", batch['input_ids'])
print("Target IDs:", batch['target_ids'])

'''second_batch = next(data_iter)
print("Second Input IDs:", second_batch['input_ids'])
print("Second Target IDs:", second_batch['target_ids'])'''
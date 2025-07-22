import tiktoken

class Tokenizer:
    def __init__(self, text):
        self.text = text
        self.tokens = []
        self.tokenizer = tiktoken.get_encoding("gpt2") #crea l'istanza di tokenizzazione utilizzando il modello GPT-2

    def encode(self):
        # Effettua l'encoding del testo in interi
        self.tokens = self.tokenizer.encode(self.text, allowed_special={"<|endoftext|>"})
        return self.tokens

    def decode(self, integer_list):
        # Decodifica la lista degli interi in testo
        return self.tokenizer.decode(integer_list)
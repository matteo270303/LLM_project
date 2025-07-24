import tiktoken

class Tokenizer:
    def __init__(self, text):
        self.text = text
        self.tokens = []
        self.tokenizer = tiktoken.get_encoding("gpt2") #crea l'istanza di tokenizzazione utilizzando il modello GPT-2

    def encode(self, text_to_encode=None):
        text = self.text if text_to_encode is None else text_to_encode
        return self.tokenizer.encode(text, allowed_special={"<|endoftext|>"})

    def decode(self, integer_list):
        # Decodifica la lista degli interi in testo
        return self.tokenizer.decode(integer_list)
    
    def vocab_len(self):
        # Restituisce la lunghezza del vocabolario
        return self.tokenizer.n_vocab
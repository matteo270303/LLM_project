#dizionario di configurazione per GPT-124M

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 512, #numero massimo di token in input che l'embedding posizionale può gestire
    "embedding_size": 768, #trasforma ogni token in un vettore di dimensione 768
    "num_layers": 2,
    "num_heads": 2,
    "drop_rate": 0.2,
    "qkv_bias": False
}
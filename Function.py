import torch
import torch.nn as nn

class GELU(nn.Module):  # Eredita da nn.Module
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh((2 / torch.pi) ** 0.5 * (x + 0.044715 * x ** 3)))
    
#Funzione generatrice per la generazione della next token prediction     
def generate_text(model, idx, max_new_tokens, context_length):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_length:]  
        
        with torch.no_grad(): 
            logits = model(idx_cond)  
        
        logits = logits[:, -1, :]  
        probas = nn.functional.softmax(logits, dim=-1)  
        idx_next = torch.argmax(probas, dim=-1, keepdim=True) 
        idx = torch.cat((idx, idx_next), dim=1)  
    return idx
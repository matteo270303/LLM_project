import torch 
from torch.utils.data import Dataset, DataLoader
import Tokenizer

class CustomDataset(Dataset):
    def __init__(self, text, tokenizer, max_length=512, stride=256):
        self.input_ids = []
        self.target_ids = []

        token_ids = tokenizer.encode()
        for i in range(0, len(token_ids) - max_length + 1, stride):
            input_ids = token_ids[i:i + max_length]
            target_ids = token_ids[i + 1:i + max_length + 1]        
            self.input_ids.append(torch.tensor(input_ids))
            self.target_ids.append(torch.tensor(target_ids))
            
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'target_ids': self.target_ids[idx]
        }

    @staticmethod
    def create_dataloader(txt, batch_size, shuffle, max_length, stride):
        tokenizer = Tokenizer.Tokenizer(txt)
        dataset = CustomDataset(txt, tokenizer, max_length=max_length, stride=stride)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
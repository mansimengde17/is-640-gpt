import torch

class Data:
    def __init__(self, file_path, device='cpu', block_size=256):
        with open(file_path, 'r', encoding='utf-8') as f:
            self.text = f.read()

        # create character-level encoding
        self.chars = sorted(list(set(self.text)))
        self.vocab_size = len(self.chars)
        self.stoi = { ch:i for i,ch in enumerate(self.chars) }
        self.itos = { i:ch for i,ch in enumerate(self.chars) }
        
        self.encode = lambda s: [self.stoi[c] for c in s]
        self.decode = lambda l: ''.join([self.itos[i] for i in l])

        self.data = torch.tensor(self.encode(self.text), dtype=torch.long)
        self.block_size = block_size
        self.device = device

    def split_data(self, split_ratio=0.9):
        n = int(split_ratio * len(self.data))
        train_data = self.data[:n]
        val_data = self.data[n:]
        return train_data, val_data

    def get_batch(self, split, batch_size, train_data, val_data):
        data = train_data if split == 'train' else val_data
        ix = torch.randint(len(data) - self.block_size, (batch_size,))
        x = torch.stack([data[i:i+self.block_size] for i in ix])
        y = torch.stack([data[i+1:i+self.block_size+1] for i in ix])
        x, y = x.to(self.device), y.to(self.device)
        return x, y
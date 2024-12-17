import torch
from data import Data
from model import GPTLanguageModel
from trainer import Trainer

# Hyperparameters
batch_size = 64
block_size = 256
max_iters = 5000
eval_interval = 200
learning_rate = 3e-4
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load data
data = Data('input.txt')
train_data, val_data = data.get_splits()

def get_batch(split):
    data_split = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_split) - block_size, (batch_size,))
    x = torch.stack([data_split[i:i+block_size] for i in ix])
    y = torch.stack([data_split[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)

# Initialize model
model = GPTLanguageModel(data.vocab_size, block_size, n_embd, n_layer, n_head, dropout).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
trainer = Trainer(model, optimizer, get_batch, device)

# Train the model
trainer.train(max_iters, eval_interval)

# Generate text
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(data.decode(model.generate(context, max_new_tokens=100)[0].tolist()))

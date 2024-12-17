"""
main.py

This script orchestrates the GPT language model training and text generation process.
It does the following accordingly:
1. Loads input data and preprocesses it.
2. Initializes the GPT model with specified hyperparameters.
3. Trains the model and periodically evaluates performance.
4. Generates new text based on the trained model.

To run: Ensure the input file 'input.txt' is in the same directory or specify the path.
"""

import torch
from data import Data
from model import GPTLanguageModel
from trainer import Trainer

# Hyperparameters: Define model training and architecture parameters
batch_size = 64 ## Number of samples processed before updating model
block_size = 256 # Length of the context window for the model
max_iters = 5000 # Total number of training iterations
eval_interval = 200 # Frequency of evaluation during training
learning_rate = 3e-4 # Learning rate for the optimizer
n_embd = 384 # Size of the token embeddings
n_head = 6 # Number of attention heads
n_layer = 6 # Number of transformer layers
dropout = 0.2 # Dropout rate for regularization

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

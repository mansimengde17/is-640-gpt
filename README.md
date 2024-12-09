# is-640-gpt
This repository contains the code for the IS640 Project 2, where we build a simplified GPT model from scratch. It includes four main parts: data.py (data handling), model.py (model architecture), trainer.py (training logic), and main.py (model execution and text generation).

GPT Language Model:

This project implements a GPT-style language model using PyTorch. It includes the core components of transformer architecture, such as self-attention, multi-head attention, feed-forward layers, and positional embeddings. The model is designed for tasks like text generation and can be easily extended or fine-tuned for other NLP applications.

Features:

Custom transformer-based architecture built from scratch.
Modular design for easy experimentation and extension.
Token embedding and positional embedding support.
Implements multi-head self-attention for contextual representation learning.
Supports autoregressive text generation with the generate method.

Quick Start

1. Clone the repository:

# git clone <your-repository-url>  
# cd <repository-folder>  

2. Install dependencies:
Make sure you have Python 3.8+ installed, then run:

# pip install torch  

3. Run the model:
Use the following code snippet to initialize and generate text with the model:


# Initialize the model  

 from model import GPTLanguageModel  
 import torch  

model = GPTLanguageModel(
    vocab_size=50257,   # Vocabulary size  
    block_size=256,     # Maximum sequence length  
    n_embd=384,         # Embedding dimension  
    n_head=6,           # Number of attention heads  
    n_layer=6,          # Number of transformer blocks  
    dropout=0.2         # Dropout rate  
)  

# Generate text  

start_idx = torch.tensor([[0]])  # Starting token  
generated = model.generate(start_idx, max_new_tokens=50)  
print("Generated Sequence:", generated)  

How It Works?

The GPTLanguageModel is a PyTorch implementation that consists of:

1. Token Embedding Layer: Converts token indices into dense vector representations.
2. Positional Embedding Layer: Adds positional information to the token embeddings.
3. Transformer Blocks: Stackable blocks containing multi-head attention and feed-forward layers.
4. Autoregressive Text Generation: Generates text one token at a time using the generate method.

Customization:

You can customize the model by changing parameters such as:

1. vocab_size: Size of the vocabulary.
2. block_size: Maximum sequence length.
3. n_embd: Dimensionality of embedding vectors.
4. n_head: Number of attention heads.
5. n_layer: Number of transformer blocks.


License
This project is licensed under the MIT License.

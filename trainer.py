import torch

class Trainer:
    def __init__(self, model, optimizer, get_batch, device):
        self.model = model
        self.optimizer = optimizer
        self.get_batch = get_batch
        self.device = device

    def train(self, max_iters, eval_interval):
        for iter in range(max_iters):
            xb, yb = self.get_batch('train')
            logits, loss = self.model(xb, yb)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            if iter % eval_interval == 0 or iter == max_iters - 1:
                print(f"Iteration {iter}, Loss: {loss.item():.4f}")

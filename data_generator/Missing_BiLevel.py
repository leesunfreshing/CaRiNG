import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple

SparseBiLevelState = namedtuple('SparseBiLevelState', ['model', 'log_scale', 'log_lambda'])

class SparseBiLevel(nn.Module):
    def __init__(self, encoder, num_ways, maxiter_inner=5, tol=1e-3):
        super().__init__()
        self.encoder = encoder
        self.num_ways = num_ways
        self.maxiter_inner = maxiter_inner
        self.tol = tol

    def column_wise_l1_norm(self, W):
        return torch.sum(torch.norm(W, p=1, dim=0))

    def loss(self, params, inputs, targets):
        logits = torch.matmul(inputs, params)
        loss = F.cross_entropy(logits, targets)
        return loss

    def adapt(self, inputs, targets):
        W = torch.zeros((inputs.shape[1], self.num_ways), requires_grad=True)
        optimizer = torch.optim.Adam([W], lr=0.01)

        for _ in range(self.maxiter_inner):
            optimizer.zero_grad()
            loss = self.loss(W, inputs, targets)
            loss.backward()
            optimizer.step()

        return W.detach()

    def outer_loss(self, params, train, test):
        train_features = self.encoder(train.inputs)
        adapted_params = self.adapt(train_features, train.targets)

        test_features = self.encoder(test.inputs)
        test_loss = self.loss(adapted_params, test_features, test.targets)
        
        # L1 regularization in the outer loop
        l1_penalty = torch.exp(params.log_lambda) * self.column_wise_l1_norm(adapted_params)
        
        # Scale the test loss
        scaled_test_loss = torch.exp(params.log_scale) * test_loss
        
        total_loss = scaled_test_loss + l1_penalty

        logs = {
            'test_loss': test_loss.item(),
            'l1_penalty': l1_penalty.item(),
            'total_loss': total_loss.item(),
            'log_scale': params.log_scale.item(),
            'log_lambda': params.log_lambda.item()
        }
        return total_loss, logs

    def forward(self, train, test):
        return self.outer_loss(self.state, train, test)

    def meta_init(self):
        self.state = SparseBiLevelState(
            model=self.encoder.state_dict(),
            log_scale=torch.tensor(0., requires_grad=True),
            log_lambda=torch.tensor(0., requires_grad=True),
        )
        return self.state
    
    
# The toy code to use the bi-level optimization

model = SparseBiLevel(encoder, num_ways)
model.meta_init()

# Outer optimization
outer_optimizer = torch.optim.Adam([
    {'params': model.encoder.parameters()},
    {'params': [model.state.log_scale, model.state.log_lambda]}
], lr=0.001)

for epoch in range(num_epochs):
    for task in task_batch:
        outer_optimizer.zero_grad()
        loss, logs = model(task.train, task.test)
        loss.backward()
        outer_optimizer.step()
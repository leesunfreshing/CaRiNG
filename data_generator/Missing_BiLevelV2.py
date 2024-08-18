import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple

SparseBiLevelState = namedtuple('SparseBiLevelState', ['model', 'log_scale', 'log_lambda'])

class SparseBiLevel(nn.Module):
    def __init__(self, encoder, num_ways, maxiter_inner=5, tol=1e-3, 
                 lambda_sparsity=0.1, lambda_align=0.1, lambda_consist=0.1, 
                 lambda_extra=0.1, delta=0.1):
        super().__init__()
        self.encoder = encoder
        self.num_ways = num_ways
        self.maxiter_inner = maxiter_inner
        self.tol = tol
        self.lambda_sparsity = lambda_sparsity
        self.lambda_align = lambda_align
        self.lambda_consist = lambda_consist
        self.lambda_extra = lambda_extra
        self.delta = delta

    def column_wise_l1_norm(self, W):
        return torch.sum(torch.norm(W, p=1, dim=0))

    def l21_norm(self, W):
        return torch.sum(torch.norm(W, p=2, dim=0))

    def l11_norm(self, W):
        return torch.sum(torch.abs(W))

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
    # Encode inputs
    train_mu, train_logvar = self.encoder(train.inputs)
    test_mu, test_logvar = self.encoder(test.inputs)
    
    # Reparameterization trick
    train_z = self.reparameterize(train_mu, train_logvar)
    test_z = self.reparameterize(test_mu, test_logvar)
    
    # Reconstruct inputs
    train_recon = self.decoder(train_z)
    test_recon = self.decoder(test_z)
    
    # Reconstruction loss
    train_recon_loss = F.mse_loss(train_recon, train.inputs)
    test_recon_loss = F.mse_loss(test_recon, test.inputs)
    
    # KL divergence
    train_kl_div = -0.5 * torch.sum(1 + train_logvar - train_mu.pow(2) - train_logvar.exp())
    test_kl_div = -0.5 * torch.sum(1 + test_logvar - test_mu.pow(2) - test_logvar.exp())
    
    # Adapt parameters
    adapted_params = self.adapt(train_z, train.targets)
    
    # Classification loss
    test_loss = self.loss(adapted_params, test_z, test.targets)
    
    # L1 regularization in the outer loop
    l1_penalty = torch.exp(params.log_lambda) * self.column_wise_l1_norm(adapted_params)
    
    # New regularization terms
    J_ft = adapted_params  # This is an approximation of J_{f,t}
    J_gt = self.encoder.weight  # Assuming the encoder is a linear layer
    
    sparsity_term = self.lambda_sparsity * (self.l21_norm(J_ft) + self.delta * self.l11_norm(J_gt))
    align_term = self.lambda_align * torch.abs(self.l11_norm(J_ft) - self.l11_norm(J_gt))
    
    if hasattr(self, 'prev_J_ft'):
        consist_term = self.lambda_consist * torch.abs(self.l11_norm(self.prev_J_ft) - self.l11_norm(J_ft))
    else:
        consist_term = 0
    self.prev_J_ft = J_ft.detach()
    
    extra_term = self.lambda_extra * self.l11_norm(J_ft)
    
    # Scale the test loss
    scaled_test_loss = torch.exp(params.log_scale) * test_loss
    
    # Combine all terms
    total_loss = (scaled_test_loss + l1_penalty + sparsity_term + align_term + consist_term + extra_term +
                  train_recon_loss + test_recon_loss + train_kl_div + test_kl_div)

    logs = {
        'test_loss': test_loss.item(),
        'train_recon_loss': train_recon_loss.item(),
        'test_recon_loss': test_recon_loss.item(),
        'train_kl_div': train_kl_div.item(),
        'test_kl_div': test_kl_div.item(),
        'l1_penalty': l1_penalty.item(),
        'sparsity_term': sparsity_term.item(),
        'align_term': align_term.item(),
        'consist_term': consist_term.item(),
        'extra_term': extra_term.item(),
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
def train_model(encoder, num_ways, num_epochs, task_batch):
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

        print(f"Epoch {epoch + 1}, Loss: {logs['total_loss']:.4f}")

    return model
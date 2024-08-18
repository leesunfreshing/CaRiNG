import torch
import torch.nn as nn
from torch.nn import functional as F
import math


class JacobianWeightedLinear(nn.Module):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input, jacobian_layer=None, mask=None):
        if jacobian_layer is None:
            jacobian_layer = self.weight.unsqueeze(0).expand(input.size(0), -1, -1)
        
        if mask is not None:
            jacobian_layer = jacobian_layer * mask.unsqueeze(1)
        
        output = torch.bmm(jacobian_layer, input.unsqueeze(-1)).squeeze(-1)
        if self.bias is not None:
            output += self.bias
        
        return output

class JacobianWeightedMLP(nn.Module):
    def __init__(self, in_features, out_features, num_layers, hidden_dim=64, bias=True):
        super().__init__()
        self.layers = nn.ModuleList()
        for l in range(num_layers):
            if l == 0:
                self.layers.append(JacobianWeightedLinear(in_features, hidden_dim, bias))
            else:
                self.layers.append(JacobianWeightedLinear(hidden_dim, hidden_dim, bias))
        self.layers.append(JacobianWeightedLinear(hidden_dim, out_features, bias))

    def forward(self, x, jacobian=None, mask=None):
        for i, layer in enumerate(self.layers):
            x, jacobian = layer(x, jacobian, mask)
            if i < len(self.layers) - 1:
                x = F.leaky_relu(x, 0.2)
                # Update jacobian for LeakyReLU
                # leaky_relu_grad = torch.where(x > 0, torch.ones_like(x), torch.full_like(x, 0.2))
                # jacobian = jacobian * leaky_relu_grad.unsqueeze(1)
        return x


class NLayerLeakyMLP(nn.Module):

    def __init__(self, in_features, out_features, num_layers, hidden_dim=64, bias=False):
        super().__init__()
        layers = [ ]
        for l in range(num_layers):
            if l == 0:
                layers.append(nn.Linear(in_features, hidden_dim))
                # layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.LeakyReLU(0.2))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
                # layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.LeakyReLU(0.2))
        layers.append(nn.Linear(hidden_dim, out_features))

        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

class MLPEncoder(nn.Module):

    def __init__(self, latent_size, num_layers=4, hidden_dim=64):
        super().__init__()
        self.net = NLayerLeakyMLP(in_features=latent_size, 
                                  out_features=latent_size, 
                                  num_layers=num_layers, 
                                  hidden_dim=hidden_dim)
    
    def forward(self, x):
        return self.net(x)

class MLPDecoder(nn.Module):
    """Ground-truth MLP decoder used for data generation"""
    def __init__(self, latent_size, num_layers=4, hidden_dim=64):
        super().__init__()
        # TODO: Do not use ground-truth decoder architecture 
        self.net = NLayerLeakyMLP(in_features=latent_size, 
                                  out_features=latent_size, 
                                  num_layers=num_layers, 
                                  hidden_dim=hidden_dim)
    
    def forward(self, z):
        return self.net(z)

class Inference(nn.Module):
    """Ground-truth MLP decoder used for data generation"""
    def __init__(self, lag, z_dim, num_layers=4, hidden_dim=64):
        super().__init__()
        self.z_dim = z_dim
        self.lag = lag
        self.f1 = nn.Linear(lag*z_dim, z_dim*2)
        self.f2 = nn.Linear(2*hidden_dim, hidden_dim)

        self.net = NLayerLeakyMLP(in_features=hidden_dim, 
                                  out_features=z_dim*2, 
                                  num_layers=num_layers, 
                                  hidden_dim=hidden_dim)

    def forward(self, x):
        zs = x[:,:self.lag*self.z_dim]
        distributions = self.f1(zs)
        enc = self.f2(x[:,self.lag*self.z_dim:])
        distributions = distributions + self.net(enc)
        return distributions

class NAC(nn.Module):    
    def __init__(self, n_in, n_out):
        super().__init__()
        self.W_hat = nn.Parameter(torch.Tensor(n_out, n_in))
        self.M_hat = nn.Parameter(torch.Tensor(n_out, n_in))
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.W_hat)         
        nn.init.kaiming_uniform_(self.M_hat)

    def forward(self, input):
        weights = torch.tanh(self.W_hat) * torch.sigmoid(self.M_hat)
        return F.linear(input, weights)

class NALU(nn.Module):    
    def __init__(self, n_in, n_out):
        super().__init__()        
        self.NAC = NAC(n_in, n_out)        
        self.G = nn.Parameter(torch.Tensor(1, n_in))        
        self.eps = 1e-6        
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.G)
    
    def forward(self, input):
        g = torch.sigmoid(F.linear(input, self.G))
        y1 = g * self.NAC(input)        
        y2 = (1 - g) * torch.exp(self.NAC(torch.log(torch.abs(input) + self.eps)))
        return y1 + y2

class NLayerLeakyNAC(nn.Module):

    def __init__(self, in_features, out_features, num_layers, hidden_dim=64, bias=True):
        super().__init__()
        layers = [ ]
        for l in range(num_layers):
            if l == 0:
                layers.append(NALU(in_features, hidden_dim))
                # layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.LeakyReLU(0.2))
            else:
                layers.append(NALU(hidden_dim, hidden_dim))
                # layers.append(nn.BatchNorm1d(hidden_dim))
                layers.append(nn.LeakyReLU(0.2))
        layers.append(NALU(hidden_dim, out_features))

        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x)

if __name__ == '__main__':
    net = NLayerLeakyMLP(3,64,32,128)
    print(net)
"""Prior Network"""
import torch
import torch.nn as nn
from torch.autograd.functional import jacobian
from functorch import jacrev, vmap
from .mlp import *
from .base import GroupLinearLayer
import ipdb as pdb


class MBDTransitionPrior(nn.Module):

    def __init__(self, lags, latent_size, bias=False):
        super().__init__()
        # self.init_hiddens = nn.Parameter(0.001 * torch.randn(lags, latent_size))    
        # out[:,:,0] = (x[:,:,0]@conv.weight[:,:,0].T)+(x[:,:,1]@conv.weight[:,:,1].T) 
        # out[:,:,1] = (x[:,:,1]@conv.weight[:,:,0].T)+(x[:,:,2]@conv.weight[:,:,1].T)
        self.L = lags      
        self.transition = GroupLinearLayer(din = latent_size, 
                                           dout = latent_size, 
                                           num_blocks = lags,
                                           diagonal = False)
        self.bias = bias
        if bias:
            self.b = nn.Parameter(0.001 * torch.randn(1, latent_size))
    
    def forward(self, x, mask=None):
        # x: [BS, T, D] -> [BS, T-L, L+1, D]
        batch_size, length, input_dim = x.shape
        # init_hiddens = self.init_hiddens.repeat(batch_size, 1, 1)
        # x = torch.cat((init_hiddens, x), dim=1)
        x = x.unfold(dimension = 1, size = self.L+1, step = 1)
        x = torch.swapaxes(x, 2, 3)
        shape = x.shape

        x = x.reshape(-1, self.L+1, input_dim)
        xx, yy = x[:,-1:], x[:,:-1]
        if self.bias:
            residuals = torch.sum(self.transition(yy), dim=1) + self.b - xx.squeeze()
        else:
            residuals = torch.sum(self.transition(yy), dim=1) - xx.squeeze()
        residuals = residuals.reshape(batch_size, -1, input_dim)
        # Dummy jacobian matrix (0) to represent identity mapping
        log_abs_det_jacobian = torch.zeros(batch_size, device=x.device)
        return residuals, log_abs_det_jacobian

class NPTransitionPrior(nn.Module):

    def __init__(
        self, 
        lags, 
        latent_size, 
        num_layers=3, 
        hidden_dim=64):
        super().__init__()
        self.L = lags
        # self.init_hiddens = nn.Parameter(0.01 * torch.randn(lags, latent_size))       
        gs = [NLayerLeakyMLP(in_features=lags*latent_size+1, 
                             out_features=1, 
                             num_layers=num_layers, 
                             hidden_dim=hidden_dim) for i in range(latent_size)]
        
        self.gs = nn.ModuleList(gs)
    
    def forward(self, x, masks=None):
        # x: [BS, T, D] -> [BS, T-L, L+1, D]
        batch_size, length, input_dim = x.shape
        # init_hiddens = self.init_hiddens.repeat(batch_size, 1, 1)
        # x = torch.cat((init_hiddens, x), dim=1)
        x = x.unfold(dimension = 1, size = self.L+1, step = 1)
        x = torch.swapaxes(x, 2, 3)
        shape = x.shape
        x = x.reshape(-1, self.L+1, input_dim)
        xx, yy = x[:,-1:], x[:,:-1]
        yy = yy.reshape(-1, self.L*input_dim)
        residuals = [ ]
        sum_log_abs_det_jacobian = 0
        for i in range(input_dim):
            if masks is None:
                inputs = torch.cat((yy, xx[:,:,i]),dim=-1)
            else:
                mask = masks[i]
                inputs = torch.cat((yy*mask, xx[:,:,i]),dim=-1)
            residual = self.gs[i](inputs)
            with torch.enable_grad():
                pdd = jacobian(self.gs[i], inputs, create_graph=True, vectorize=True)
            # Determinant of low-triangular mat is product of diagonal entries
            logabsdet = torch.log(torch.abs(torch.diag(pdd[:,0,:,-1])))
            sum_log_abs_det_jacobian += logabsdet
            residuals.append(residual)

        residuals = torch.cat(residuals, dim=-1)
        residuals = residuals.reshape(batch_size, -1, input_dim)
        sum_log_abs_det_jacobian = torch.sum(sum_log_abs_det_jacobian.reshape(batch_size, length-self.L), dim=1)
        return residuals, sum_log_abs_det_jacobian
    

class NPSparseTransitionPrior(nn.Module):
    def __init__(
        self, 
        lags, 
        latent_size, 
        num_layers=3, 
        hidden_dim=64,
        epsilon=1e-6):
        super().__init__()
        self.L = lags
        self.latent_size = latent_size
        self.epsilon = epsilon
        
        gs = [NLayerLeakyMLP(in_features=lags*latent_size+1, 
                             out_features=1, 
                             num_layers=num_layers, 
                             hidden_dim=hidden_dim) for i in range(latent_size)]
        
        # gs = [NLayerLeakyMLPWithMissingData(in_features=lags*latent_size+1, 
        #                                     out_features=1, 
        #                                     num_layers=num_layers, 
        #                                     hidden_dim=hidden_dim) for i in range(latent_size)]
        
        self.gs = nn.ModuleList(gs)
    
    def forward(self, x, masks=None):        
        # x: [BS, T, D] -> [BS, T-L, L+1, D]
        # embeddings: [BS, embed_dims]
        batch_size, length, input_dim = x.shape
        
        # mask = torch.ones_like(x)
        
        x = x.unfold(dimension=1, size=self.L+1, step=1)
        x = torch.swapaxes(x, 2, 3)
        x = x.reshape(-1, self.L+1, input_dim)
        xx, yy = x[:,-1:], x[:,:-1]
        yy = yy.reshape(-1, self.L*input_dim)
        residuals = []
        sum_log_abs_det_jacobian = 0
        transition_jacobians = []
        
        # "mask the parents"
        # for t in range(length):
        #     if t < 3:
        #         mask[:, t, 0] = 0
        #     elif t >= 3 and t < 6:
        #         mask[:, t, 2] = 0
        #     else:
        #         mask[:, t, 1] = 0
        # mask = mask.unfold(dimension=1, size=self.L+1, step=1)
        # mask = torch.swapaxes(mask, 2, 3)
        # mask = mask.reshape(-1, self.L+1, input_dim)
        # xx_mask, yy_mask = mask[:,-1:], mask[:,:-1]
        # yy_mask = yy_mask.reshape(-1, self.L*input_dim)

        for i in range(input_dim):
            if masks is None:
                inputs = torch.cat((yy, xx[:,:,i]), dim=-1)
                # inputs = torch.cat((yy*yy_mask, xx[:,:,i]*xx_mask[:,:,i]), dim=-1)
            else:
                mask = masks[i]
                inputs = torch.cat((yy*mask, xx[:,:,i]), dim=-1)
            
            # residual = self.gs[i](inputs)
            
            residual, jac = vmap(self.compute_residual_and_jacobian, in_dims=(None, 0))(self.gs[i], inputs)
            
            # Compute log abs det Jacobian for change of variables
            logabsdet = torch.log(torch.abs(jac[:, -1] + self.epsilon))
            sum_log_abs_det_jacobian += logabsdet
            
            # Compute transition Jacobian with implicit function theorem
            d_residual_d_yy = jac[:, :-1]  # Jacobian w.r.t yy (previous states)
            d_residual_d_xx = jac[:, -1:]  # Jacobian w.r.t xx (current state)

            transition_jac = -d_residual_d_yy / (d_residual_d_xx + self.epsilon)
            
            transition_jacobians.append(transition_jac)
            
            residuals.append(residual)

        residuals = torch.cat(residuals, dim=-1)
        residuals = residuals.reshape(batch_size, -1, input_dim)
        sum_log_abs_det_jacobian = torch.sum(sum_log_abs_det_jacobian.reshape(batch_size, length-self.L), dim=1)
        
        # Stack transition Jacobians with size of [batch_size, length-self.L, input_dim, input_dim]
        transition_jacobian = torch.stack(transition_jacobians, dim=1).reshape(batch_size, length-self.L, input_dim, input_dim)
        
        # Compute L1 norms for xx and yy, the minimum row / col should be missing?
        xx_spa = []
        yy_spa = []
        for t in range(length-self.L):
            xx_spa.append(torch.norm(transition_jacobian[:,t], p=0, dim=-2).mean())
            if t >= 1:
                yy_spa.append(torch.norm(transition_jacobian[:,t], p=0, dim=-1).mean())
        
        return residuals, sum_log_abs_det_jacobian, transition_jacobian, xx_spa, yy_spa
    
    def compute_residual_and_jacobian(self, g, inputs):
        def g_wrapper(inputs):
            return g(inputs).squeeze(-1)
        
        jac_fn = jacrev(g_wrapper)
        residual = g_wrapper(inputs)
        jac = jac_fn(inputs)
        return residual, jac
    

class NPChangeTransitionPrior(nn.Module):

    def __init__(
        self, 
        lags, 
        latent_size,
        embedding_dim, 
        num_layers=3,
        hidden_dim=64):
        super().__init__()
        self.L = lags
        # self.init_hiddens = nn.Parameter(0.01 * torch.randn(lags, latent_size))       
        gs = [NLayerLeakyMLP(in_features=hidden_dim+lags*latent_size+1, 
                             out_features=1, 
                             num_layers=num_layers, 
                             hidden_dim=hidden_dim) for i in range(latent_size)]
        
        self.gs = nn.ModuleList(gs)
        self.fc = NLayerLeakyMLP(in_features=embedding_dim,
                                 out_features=hidden_dim,
                                 num_layers=2,
                                 hidden_dim=hidden_dim)

    def forward(self, x, embeddings, masks=None):
        # x: [BS, T, D] -> [BS, T-L, L+1, D]
        # embeddings: [BS, embed_dims]
        batch_size, length, input_dim = x.shape
        embeddings = self.fc(embeddings)
        # init_hiddens = self.init_hiddens.repeat(batch_size, 1, 1)
        # x = torch.cat((init_hiddens, x), dim=1)
        x = x.unfold(dimension = 1, size = self.L+1, step = 1)
        x = torch.swapaxes(x, 2, 3)
        shape = x.shape
        x = x.reshape(-1, self.L+1, input_dim)
        xx, yy = x[:,-1:], x[:,:-1]
        yy = yy.reshape(-1, self.L*input_dim)
        residuals = [ ]
        sum_log_abs_det_jacobian = 0
        for i in range(input_dim):
            if masks is None:
                inputs = torch.cat((embeddings, yy, xx[:,:,i]),dim=-1)
            else:
                mask = masks[i]
                inputs = torch.cat((embeddings, yy*mask, xx[:,:,i]),dim=-1)
            residual = self.gs[i](inputs)
            with torch.enable_grad():
                pdd = jacobian(self.gs[i], inputs, create_graph=True, vectorize=True)
            # Determinant of low-triangular mat is product of diagonal entries
            logabsdet = torch.log(torch.abs(torch.diag(pdd[:,0,:,-1])))
            sum_log_abs_det_jacobian += logabsdet
            residuals.append(residual)

        residuals = torch.cat(residuals, dim=-1)
        residuals = residuals.reshape(batch_size, -1, input_dim)
        sum_log_abs_det_jacobian = torch.sum(sum_log_abs_det_jacobian.reshape(batch_size, length-self.L), dim=1)
        return residuals, sum_log_abs_det_jacobian

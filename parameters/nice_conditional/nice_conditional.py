"""Utility classes for NICE.
"""

import torch
import torch.nn as nn
import numpy as np

"""Additive coupling layer.
"""
class Coupling(nn.Module):
    def __init__(self, len_input, mid_dim, hidden, hidden_time_block, mask_config, time_dim, hidden_time_dim = 5, out_time_dim = 5):
        """Initialize a coupling layer.
        Args:
            len_input: input/output dimensions.
            mid_dim: number of units in a hidden layer.
            hidden: number of hidden layers.
            mask_config: 1 if transform odd units, 0 if transform even units.
        """
        super(Coupling, self).__init__()
        self.mask_config = mask_config

        self.in_time_in_block = nn.Sequential(
            nn.Linear(time_dim, hidden_time_dim),
            )  
            # nn.ReLU())
        self.in_time_mid_block = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_time_dim, hidden_time_dim),
                ) for _ in range(hidden_time_block - 1)])
                # nn.ReLU()) for _ in range(hidden_time_block - 1)])
        self.in_time_out_block = nn.Linear(hidden_time_dim, out_time_dim)

        self.in_block = nn.Sequential(
            nn.Linear(len_input//2 + out_time_dim, mid_dim),
            nn.ReLU())
        self.mid_block = nn.ModuleList([
            nn.Sequential(
                nn.Linear(mid_dim, mid_dim),
                nn.ReLU()) for _ in range(hidden - 1)])
        self.out_block = nn.Linear(mid_dim, len_input//2)
        self.len_input = len_input

    def forward(self, z, x, reverse=False):
        """Forward pass.
        Args:
            x: input tensor.
            reverse: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor.
        """
        [B, W] = list(z.size())
        z = z.reshape((B, W//2, 2))
        if self.mask_config:
            on, off = z[:, :, 0], z[:, :, 1]
        else:
            off, on = z[:, :, 0], z[:, :, 1]
        
        off_time = torch.sin(self.in_time_in_block(x))
        for i in range(len(self.in_time_mid_block)):
            off_time = torch.sin(self.in_time_mid_block[i](off_time))
        off_time = self.in_time_out_block(off_time)
        
        off_concatenated = torch.cat((off,off_time),axis = -1)
        off_ = self.in_block(off_concatenated)
        for i in range(len(self.mid_block)):
            off_ = self.mid_block[i](off_)
        shift = self.out_block(off_)
        if reverse:
            on = on - shift
        else:
            on = on + shift

        if self.mask_config:
            z = torch.stack((on, off), dim=2)
        else:
            z = torch.stack((off, on), dim=2)
        return z.reshape((B, W))

"""Log-scaling layer.
"""
class Scaling(nn.Module):
    def __init__(self, dim):
        """Initialize a (log-)scaling layer.
        Args:
            dim: input/output dimensions.
        """
        super(Scaling, self).__init__()
        self.scale = nn.Parameter(
            torch.zeros((1, dim)), requires_grad=True)

    def forward(self, x, reverse=False):
        """Forward pass.
        Args:
            x: input tensor.
            reverse: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor and log-determinant of Jacobian.
        """
        log_det_J = torch.sum(self.scale)
        if reverse:
            x = x * torch.exp(-self.scale)
        else:
            x = x * torch.exp(self.scale)
        return x, log_det_J

class Conditional_prior(nn.Module):
    def __init__(self, mid_dim, out_dim, hidden = 3,time_dim = 13):
        """Initialize a (log-)scaling layer.
        Args:
            dim: input/output dimensions.
        """
        super(Conditional_prior, self).__init__()

        self.in_block_mean = nn.Sequential(
            nn.Linear(time_dim, mid_dim),
            )
            # nn.ReLU())
        self.mid_block_mean = nn.ModuleList([
            nn.Sequential(
                nn.Linear(mid_dim, mid_dim),
                ) for _ in range(hidden - 1)])
                # nn.ReLU()) for _ in range(hidden - 1)])
        self.out_block_mean = nn.Linear(mid_dim, out_dim)

        self.in_block_var = nn.Sequential(
            nn.Linear(time_dim, mid_dim),
            )
            # nn.ReLU())
        self.mid_block_var = nn.ModuleList([
            nn.Sequential(
                nn.Linear(mid_dim, mid_dim),
                ) for _ in range(hidden - 1)])
                # nn.ReLU()) for _ in range(hidden - 1)])
        self.out_block_var = nn.Linear(mid_dim, out_dim)



    def forward(self, z, x, reverse=False):
        """Forward pass.
        Args:
            x: input tensor.
            reverse: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor and log-determinant of Jacobian.
        """
        off_ = torch.sin(self.in_block_mean(x))
        for i in range(len(self.mid_block_var)):
            off_ = torch.sin(self.mid_block_mean[i](off_))
        mean = self.out_block_mean(off_)

        off_ = torch.sin(self.in_block_var(x))
        for i in range(len(self.mid_block_var)):
            off_ = torch.sin(self.mid_block_var[i](off_))
        var = self.out_block_var(off_)

        
        log_det_J_p_1 = torch.log(torch.abs(1/torch.prod(var,axis = 1)))
        
        if reverse:
            return z*var + mean, log_det_J_p_1

        else:
            return (z - mean)/var, log_det_J_p_1


"""NICE main model.
"""
class NICE_CONDITIONAL(nn.Module):
    def __init__(self, prior, coupling, 
        len_input, mid_dim, hidden, mask_config,time_dim):
        """Initialize a NICE.
        Args:
            prior: prior distribution over latent space Z.
            coupling: number of coupling layers.
            len_input: input/output dimensions.
            mid_dim: number of units in a hidden layer.
            hidden: number of hidden layers.
            mask_config: 1 if transform odd units, 0 if transform even units.
        """
        mid_time_dim = 7
        hidden_block_time = 10
        super(NICE_CONDITIONAL, self).__init__()
        self.prior = prior
        self.len_input = len_input

        self.coupling = nn.ModuleList([
            Coupling(len_input=len_input, 
                     mid_dim=mid_dim, 
                     hidden=hidden,
                     hidden_time_block = hidden_block_time,
                     mask_config=(mask_config+i)%2,
                     time_dim=time_dim,
                     hidden_time_dim = mid_time_dim, 
                     out_time_dim = 5) \
            for i in range(coupling)])
        self.scaling = Scaling(len_input)

        self.conditional_prior = Conditional_prior(mid_dim = 5, out_dim = len_input, hidden=hidden_block_time, time_dim = time_dim)

    def g(self, z, time):
        """Transformation g: Z -> X (inverse of f).
        Args:
            z: tensor in latent space Z.
        Returns:
            transformed tensor in data space X.
        """
        ####
        z, _ = self.conditional_prior(z, time, reverse = True)
        ####
        z, _ = self.scaling(z, reverse=True)
        for i in reversed(range(len(self.coupling))):
            z = self.coupling[i](z, time, reverse=True)
        return z

    def f(self, z, x):
        """Transformation f: X -> Z (inverse of g).
        Args:
            x: tensor in data space X.
        Returns:
            transformed tensor in latent space Z.
        """
        for i in range(len(self.coupling)):
            z = self.coupling[i](z,x)
        return self.scaling(z)

    def log_prob(self, temperature, time):
        """Computes data log-likelihood.
        (See Section 3.3 in the NICE paper.)
        Args:
            x: input minibatch.
        Returns:
            log-likelihood of input.
        """
        z, log_det_J = self.f(temperature, time)
        ####
        z, log_det_J_p_1 = self.conditional_prior(z, time, reverse = False)
        ####

        prob_posteriori = torch.exp(-torch.square(z)/2 ) + 1e-6
        # print(prob_posteriori)
        # raise NotImplementedError
        log_prob_posteriori = torch.log( prob_posteriori )
        log_ll = torch.sum(log_prob_posteriori,dim = 1)

        log_ll = torch.sum(self.prior.log_prob(z), dim=1)
        # log_ll = torch.sum(log_prob_posteriori,dim = 1)
        return log_ll + log_det_J + log_det_J_p_1

    def sample(self, size):
        """Generates samples.
        Args:
            size: number of samples to generate.
        Returns:
            samples from the data space X.
        """
        z = self.prior.sample((size, self.len_input)).cuda()
        return self.g(z)

    def forward(self, temperature, time):
        """Forward pass.
        Args:
            x: input minibatch.
        Returns:
            log-likelihood of input.
        """
        return self.log_prob(temperature, time)
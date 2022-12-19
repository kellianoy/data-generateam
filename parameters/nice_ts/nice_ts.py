"""Utility classes for NICE.
"""

import torch
import torch.nn as nn

"""Additive coupling layer.
"""
class Coupling(nn.Module):
    def __init__(self, len_input, mid_dim, hidden, mask_config,len_rnn_embedding_input):
        """Initialize a coupling layer.
        Args:
            len_input: input/output dimensions.
            mid_dim: number of units in a hidden layer.
            hidden: number of hidden layers.
            mask_config: 1 if transform odd units, 0 if transform even units.
        """
        super(Coupling, self).__init__()
        self.mask_config = mask_config

        self.in_block = nn.Sequential(
            nn.Linear(len_input//2 + len_rnn_embedding_input, mid_dim),
            nn.ReLU())
        self.mid_block = nn.ModuleList([
            nn.Sequential(
                nn.Linear(mid_dim, mid_dim),
                nn.ReLU()) for _ in range(hidden - 1)])
        self.out_block = nn.Linear(mid_dim, len_input//2)
        self.len_input = len_input

    def forward(self, x, rnn_embedding ,reverse=False):
        """Forward pass.
        Args:
            x: input tensor.
            reverse: True in inference mode, False in sampling mode.
        Returns:
            transformed tensor.
        """
        [B, W] = list(x.size())
        x = x.reshape((B, W//2, 2))
        if self.mask_config:
            on, off = x[:, :, 0], x[:, :, 1]
        else:
            off, on = x[:, :, 0], x[:, :, 1]

        off_ = self.in_block(torch.cat( ( off, rnn_embedding),axis = -1 ))

        for i in range(len(self.mid_block)):
            off_ = self.mid_block[i](off_)
        shift = self.out_block(off_)
        if reverse:
            on = on - shift
        else:
            on = on + shift

        if self.mask_config:
            x = torch.stack((on, off), dim=2)
        else:
            x = torch.stack((off, on), dim=2)
        return x.reshape((B, W))

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


class RNN(nn.Module):
    def __init__(self, input_size, embedding_dim):
        super(RNN, self).__init__()
        self.lstm = nn.LSTM(input_size, embedding_dim, batch_first=True, bidirectional=False)

    def froward_with_hidden(self, x, time, hidden):
        x = torch.cat((x,time),axis = -1)
        out, hidden = self.lstm(x, hidden)
        return out, hidden

    def forward(self, x, time):
        x = torch.cat((x,time),axis = -1)
        out, _ = self.lstm(x)
        rnn_embedding = torch.squeeze(out[:,-1,:])
        return rnn_embedding


"""NICE main model.
"""
class NICE_TS(nn.Module):
    def __init__(self, prior, coupling, 
        len_input, mid_dim, hidden, mask_config, rnn_embedding_dim, len_input_rnn):
        """Initialize a NICE.
        Args:
            prior: prior distribution over latent space Z.
            coupling: number of coupling layers.
            len_input: input/output dimensions.
            mid_dim: number of units in a hidden layer.
            hidden: number of hidden layers.
            mask_config: 1 if transform odd units, 0 if transform even units.
        """
        super(NICE_TS, self).__init__()
        self.prior = prior
        self.len_input = len_input

        self.coupling = nn.ModuleList([
            Coupling(len_input=len_input, 
                     mid_dim=mid_dim, 
                     hidden=hidden, 
                     mask_config=(mask_config+i)%2,
                     len_rnn_embedding_input = rnn_embedding_dim) \
            for i in range(coupling)])
        self.scaling = Scaling(len_input)

        ####Complete later
        self.rnn = RNN(input_size = len_input_rnn, 
                       embedding_dim = rnn_embedding_dim)
        self.embedding_dim = rnn_embedding_dim
        ######

    def g(self, z, temperature, t):
        """Transformation g: Z -> X (inverse of f).
        Args:
            z: tensor in latent space Z.
        Returns:
            transformed tensor in data space X.
        """

        rnn_embedding = self.rnn(temperature,t)
        if rnn_embedding.dim() == 1:
            rnn_embedding = rnn_embedding.unsqueeze(0)

        x, _ = self.scaling(z, reverse=True)
        for i in reversed(range(len(self.coupling))):
            x = self.coupling[i](x, rnn_embedding.squeeze(1),reverse=True)
        return x

    def f(self, x, t):
        """Transformation f: X -> Z (inverse of g).
        Args:
            x: tensor in data space X.
        Returns:
            transformed tensor in latent space Z.
        """
        input_NVP = torch.squeeze(x[:,-1,:])

        rnn_embedding = self.rnn(x, t)

        for i in range(len(self.coupling)):
            input_NVP = self.coupling[i](input_NVP, rnn_embedding)
        return self.scaling(input_NVP)

    def log_prob(self, x, time):
        """Computes data log-likelihood.
        (See Section 3.3 in the NICE paper.)
        Args:
            x: input minibatch.
        Returns:
            log-likelihood of input.
        """
        z, log_det_J = self.f(x, time)
        ####
        # torch.pi = torch.acos(torch.zeros(1)).item() * 2
        # time_mean = torch.stack([time for i in range(0,self.len_input)], dim=1)
        # prob_posteriori = torch.exp(-torch.square(z - time_mean)/( 2*(self.prior)**2 ) )/(2*torch.pi*self.prior)
        # log_prob_posteriori = torch.log( prob_posteriori )
        ####
        log_ll = torch.sum(self.prior.log_prob(z), dim=1)
        # log_ll = torch.sum(log_prob_posteriori,dim = 1)
        return log_ll + log_det_J

    def sample(self, size):
        """Generates samples.
        Args:
            size: number of samples to generate.
        Returns:
            samples from the data space X.
        """
        z = self.prior.sample((size, self.len_input)).cuda()
        return self.g(z)

    def forward(self, x, time):
        """Forward pass.
        Args:
            x: input minibatch.
        Returns:
            log-likelihood of input.
        """
        return self.log_prob(x, time)
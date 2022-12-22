import torch
from torch import nn
from parameters.glow import Glow

class Trainer():
    def __init__(self, model, lr=0.1):
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(model.parameters(),lr=lr)
        self.model = model

    def training_iteration(self, temperature, time):
        raise NotImplementedError
        return loss

    def generate_sample(self, n_sample, time_interval):
        noise = torch.empty(( n_sample, self.model.len_input)).normal_(mean=0,std=1)
        time = torch.FloatTensor(n_sample).uniform_(time_interval[0], time_interval[1])

        with torch.no_grad():
            return self.model.g(noise,time).cpu().detach().numpy()
    
    def model_to_save(self):
        return self.model
    
    def load_model(self,PATH):
        checkpoint = torch.load(PATH)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        return self.model
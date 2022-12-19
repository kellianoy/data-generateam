import torch
from torch import nn
from parameters.nice import NICE
import numpy as np
from data.dataset_tools import get_month_from_scaled_float

class Trainer():
    def __init__(self, model, lr=0.1):
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(model.parameters(),lr=lr)
        self.model = model

    def training_iteration(self, temperature, time):
        self.optimizer.zero_grad()
        loss = -self.model(temperature,time).mean()
        loss.backward()
        self.optimizer.step()
        return loss

    def generate_sample(self, n_sample, time_interval):
        noise = torch.empty(( n_sample, self.model.len_input)).normal_(mean=0,std=1)
        time_y = torch.FloatTensor(n_sample).uniform_(time_interval[0][0], time_interval[1][0])
        month = get_month_from_scaled_float(time_y)
        time = torch.zeros((n_sample,time_interval[0].shape[0]), dtype=torch.float32)
        time[:,0] = time_y
        for i in range(n_sample):
            time[i,month[i]] = 1

        # noise = torch.empty(( n_sample, self.model.len_input))
        # for i in range(n_sample):
        #     noise = noise.normal_(mean=time[i,0],std=self.model.prior)
        with torch.no_grad():
            return self.model.g(noise,time).cpu().detach().numpy()
    
    def model_to_save(self):
        return self.model
    
    def load_model(self,PATH):
        checkpoint = torch.load(PATH)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        return self.model
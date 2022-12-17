import torch
from torch import nn
from parameters.nice_ts import NICE_TS
import random

class Trainer():
    def __init__(self, model, lr=0.1):
        self.criterion = nn.BCELoss()
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.model = model

    def training_iteration(self, temperatures, times):
        self.optimizer.zero_grad()
        # input = temperature
        loss = -self.model(temperatures, times).mean()
        loss.backward()
        self.optimizer.step()
        return loss

    def generate_sample(self, n_sample, initial_temperature, time, past_infos, number_ts):
        with torch.no_grad():
            past_temperature = past_infos[0].repeat(number_ts,1,1)
            past_time = past_infos[1].repeat(number_ts,1,1)
            time = torch.tensor(time, dtype=torch.float32)


            len_sequence = time.shape[0]
            generated_sample = torch.zeros((number_ts, len_sequence, len(initial_temperature)))
            noise = torch.empty(( number_ts ,len_sequence, len(initial_temperature))).normal_(mean=0,std=1)
            for i in range(len_sequence):
                temperature = self.model.g(noise[:,i,:],
                                            past_temperature, 
                                            past_time) 

                generated_sample[:,i,:] = temperature

                past_temperature = torch.cat([past_temperature[:,1:,:], temperature.unsqueeze(1)], axis = 1)
                past_time = torch.cat( [past_time[:,1:,:], time[i].repeat(number_ts,1).unsqueeze(1)], axis = 1 )


        indices = torch.randperm(generated_sample.shape[1])[:n_sample]
        selected_sample = generated_sample[:,indices,:]
        return torch.mean(selected_sample, axis = 0).cpu().detach().numpy()

    def model_to_save(self):
        return self.model

    def load_model(self, PATH):
        checkpoint = torch.load(PATH)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        return self.model

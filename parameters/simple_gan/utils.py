import torch
from torch import nn
from parameters.simple_gan import Generator, Model
import numpy as np


class Trainer():
    def __init__(self, model, lr=0.1):
        self.criterion = nn.BCELoss()
        self.optimizerG = torch.optim.Adam(model.generator.parameters(),lr=lr)
        self.optimizerD = torch.optim.Adam(model.generator.parameters(),lr=lr)
        self.model = model

    def weights_init_uniform_rule(self,model):
        classname = model.__class__.__name__
        # for every Linear layer in a model..
        if classname.find('Linear') != -1:
            # get the number of the inputs
            n = model.in_features
            y = 1.0/np.sqrt(n)
            model.weight.data.uniform_(-y, y)
            model.bias.data.fill_(0)

    def training_iteration(self, temperature , time):

        for param in self.model.generator.parameters():
            param.requires_grad = True
        for param in self.model.discriminator.parameters():
            param.requires_grad = True

        batch_size = time.shape[0]


        #### training discriminator
        self.optimizerD.zero_grad()

        self.model.discriminator.train()
        self.model.generator.eval()

        noise = torch.empty((batch_size,self.model.len_input_noise)).normal_(mean=0,std=1)
        input = torch.cat([noise,time.unsqueeze(1)],axis = -1)
        sample_generated = self.model.generator(input)

        discriminator_input = torch.cat([sample_generated,time.unsqueeze(1)],axis = -1)
        discriminator_output = self.model.discriminator(discriminator_input)
        loss_fake_sample = self.criterion(discriminator_output, torch.ones_like(discriminator_output))

        real_sample = torch.cat([temperature,time.unsqueeze(1)],axis = -1)
        discriminator_output = self.model.discriminator(real_sample)
        loss_real_sample = self.criterion(discriminator_output, torch.zeros_like(discriminator_output))
        discriminator_loss = loss_real_sample + loss_fake_sample

        discriminator_loss.backward()
        self.optimizerD.step()



        ### training generator
        self.optimizerG.zero_grad()

        self.model.discriminator.eval()
        self.model.generator.train()

        noise = torch.empty((batch_size,self.model.len_input)).normal_(mean=1,std=1)
        input = torch.cat([noise,time.unsqueeze(1)],axis = -1)
        sample_generated = self.model.generator(input)

        discriminator_input = torch.cat([sample_generated,time.unsqueeze(1)],axis = -1)
        discriminator_output = self.model.discriminator(discriminator_input)
        generator_loss = self.criterion(discriminator_output, torch.zeros_like(discriminator_output))

        generator_loss.backward()
        self.optimizerG.step()

        losses = [discriminator_loss.item(),generator_loss.item()]
        return losses

    def generate_sample(self, n_sample , time_interval):
        for param in self.model.generator.parameters():
            param.requires_grad = False

        noise = torch.empty(( n_sample, self.model.len_input)).normal_(mean=0,std=1)
        time = torch.FloatTensor(n_sample, 1).uniform_(time_interval[0], time_interval[1])

        input_generator = torch.cat([noise,time],axis = -1)

        sample_generated = self.model.generator(input_generator)


        return sample_generated.cpu().detach().numpy()

    

    def model_to_save(self):
        return self.model.generator


    def load_model(self, PATH):
        checkpoint = torch.load(PATH)
        self.model.generator.load_state_dict(checkpoint['model_state_dict'])
        return self.model

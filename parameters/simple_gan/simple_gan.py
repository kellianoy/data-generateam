import torch
from torch import nn


class Generator(nn.Module):
    def __init__(self, len_input):
        super(Generator, self).__init__()
        self.activation = torch.nn.ReLU()

        self.linear1 = torch.nn.Linear(len_input + 1, 10)
        self.linear2 = torch.nn.Linear(10, 10)
        self.linear3 = torch.nn.Linear(10, 10)
        self.linear4 = torch.nn.Linear(10, 10)

        self.dropout = nn.Dropout(p=0.5)

        

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)

        x = self.linear2(x)
        x = self.dropout(x)
        x = self.activation(x)

        x = self.linear3(x)
        x = self.dropout(x)
        x = self.activation(x)

        x = self.linear4(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.activation = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()
        self.dropout = nn.Dropout(p=0.5)

        self.linear1 = torch.nn.Linear(11, 10)
        self.linear2 = torch.nn.Linear(10, 10)
        self.linear3 = torch.nn.Linear(10, 10)
        self.linear4 = torch.nn.Linear(10, 1)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)

        x = self.linear2(x)
        x = self.dropout(x)
        x = self.activation(x)

        x = self.linear3(x)
        x = self.dropout(x)
        x = self.activation(x)

        x = self.linear4(x)
        x = self.sigmoid(x)
        return x

class Model():
    def __init__(self,len_input):
        super(Model, self).__init__()
        self.generator = Generator(len_input)
        self.discriminator = Discriminator()
        self.len_input = len_input

    def generate(self,input):
        return self.generator(input)
    

    # def learning(self,gd):
    #     if gd:
    #         self.generator.features.parameters().requires_grad = True
    #         self.discriminator.parameters().requires_grad = True
    #     else:
    #         self.generator.parameters().requires_grad = False
    #         self.discriminator.parameters().requires_grad = False
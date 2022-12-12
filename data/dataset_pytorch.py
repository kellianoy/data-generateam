import torch
import numpy as np

class Dataset(torch.utils.data.Dataset):
  def __init__(self, temperature_array, time_array):
        self.temperature_array = temperature_array
        self.time_array = time_array
        self.len = time_array.shape[0]

  def __len__(self):
        return self.len

  def __getitem__(self, index):

        temperature = self.temperature_array[index]
        time = self.time_array[index]

        return temperature, time
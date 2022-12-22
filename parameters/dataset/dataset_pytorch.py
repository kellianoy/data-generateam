import torch

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

class Timeseries_dataset(torch.utils.data.Dataset):
  def __init__(self, temperature_array, time_array, memory_size):
        self.temperature_array = temperature_array
        self.time_array = time_array
        self.len = time_array.shape[0]
        self.memory_size = memory_size

  def __len__(self):
        return self.len

  def __getitem__(self, index):

      index_indetentation = index + 1
        
      if index >= self.memory_size:
            temperatures = self.temperature_array[index_indetentation-self.memory_size:index_indetentation]
            times = self.time_array[index_indetentation-self.memory_size:index_indetentation]
            return temperatures, times
      else:
            temperatures_empty = torch.zeros(( self.memory_size - index_indetentation ,self.temperature_array.size(1)))
            times_empty = torch.zeros(( self.memory_size - index_indetentation , self.time_array.size(1)))
            temperatures = self.temperature_array[0:index_indetentation]
            times = self.time_array[0:index_indetentation]
            temperatures_filled = torch.cat( (temperatures_empty, temperatures), axis = 0 )
            times_filled = torch.cat( (times_empty, times), axis = 0 )
            return temperatures_filled, times_filled
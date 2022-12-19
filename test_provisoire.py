from torch import nn
import torch
import numpy as np

lstm = nn.LSTM(3, 10)  # Input dim is 3, output dim is 3
inputs = [torch.randn(25,1, 3) for _ in range(5)]  # make a sequence of length 5

# initialize the hidden state.
hidden = (torch.randn(25, 1, 10),torch.randn(25, 1, 10))
for i in inputs:

    # Step through the sequence one element at a time.
    # after each step, hidden co
    # ntains the hidden state.
    print(i.shape)
    out, hidden = lstm(i, hidden)
    print("out",out.shape)
# alternatively, we can do the entire sequence all at once.
# the first value returned by LSTM is all of the hidden states throughout
# the sequence. the second is just the most recent hidden state
# (compare the last slice of "out" with "hidden" below, they are the same)
# The reason for this is that:
# "out" will give you access to all hidden states in the sequence
# "hidden" will allow you to continue the sequence and backpropagate,
# by passing it as an argument  to the lstm at a later time
# Add the extra 2nd dimension
inputs = torch.cat(inputs).view(len(inputs), 1, -1)
print(inputs.size())
hidden = (torch.randn(1, 1, 10), torch.randn(1, 1, 10))  # clean out hidden state
# print(hidden)
out, hidden = lstm(inputs, hidden)
print(out.size())
print(hidden)

A = np.array([1,2,3,4,5])
print(A[0:1])
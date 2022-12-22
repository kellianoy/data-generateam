#############################################################################
# YOUR GENERATIVE MODEL
# ---------------------
# Should be implemented in the 'generative_model' function
# !! *DO NOT MODIFY THE NAME OF THE FUNCTION* !!
#
# You can store your parameters in any format you want (npy, h5, json, yaml, ...)
# <!> *SAVE YOUR PARAMETERS IN THE parameters/ DIRECTORY* <!>
#
# See below an example of a generative model
# G_\theta(Z) = np.max(0, \theta.Z)
############################################################################

import numpy as np
import os
import torch
from parameters.nice_conditional import NICE_CONDITIONAL
from parameters.nice_conditional.utils import Trainer
from parameters.dataset.dataset_tools import get_month_from_scaled_float

# <!> DO NOT ADD ANY OTHER ARGUMENTS <!>


def generative_model(noise):
    """
    Generative model

    Parameters
    ----------
    noise : ndarray with shape (n_samples, n_dim)
        input noise of the generative model
    """
    # Forcing the type of the noise to float32 to avoid errors with torch
    noise = noise.astype('float32')

    # Setting the number of samples & the number of dimensions of the noise
    n_samples = noise.shape[0]
    len_dim = 10

    # Use the first 10 dimensions of the noise & convert latent_variable to a torch tensor
    latent_variable = torch.from_numpy(noise[:, :len_dim])

    # Define the time period of the total data
    # The first value corresponds to 1981-09-01 and the second value corresponds to 2017-01-01
    time_period = (1981.6657534246576, 2017)

    # Normalize the time period
    start_time = 2008
    normalized_start_time = (
        start_time - time_period[0]) / (time_period[1] - time_period[0])

    # Time vector (linear space of the time period we're trying to predict)
    time_vector = torch.FloatTensor(
        n_samples).uniform_(normalized_start_time, 1)

    # Get the months from the time vector
    month = get_month_from_scaled_float(time_vector)

    # Append the time vector with months
    time = torch.zeros((n_samples, 13), dtype=torch.float32)

    # One hot encoding of months
    time[:, 0] = time_vector
    for i in range(n_samples):
        time[i, month[i]] = 1

    # Generate a noise input for the model with a normal distribution
    noise_input = torch.distributions.Normal(
        torch.tensor(0.), torch.tensor(1.))

    # Create the structure of the model (NICE CONDITIONAL)
    model = NICE_CONDITIONAL(prior=noise_input,
                             coupling=6,
                             len_input=len_dim,
                             mid_dim=15,
                             hidden=4,
                             mask_config=1,
                             time_dim=13,
                             mid_time_dim=9,
                             number_hidden_block_time=9)
    trainer = Trainer(model)

    # Load the model
    model_trained = trainer.load_model(
        "parameters/nice_conditional/models_saved/optimal_model.pt")

    # Set the model to evaluation mode
    model_trained.eval()

    # Convert the numpy array to a torch tensor
    output = model_trained.g(
        latent_variable, time).cpu().detach().numpy()

    # As the network is trained to output data normalized between -1/2 and 1/2, we need to rescale the data.
    # These values correspond to the minimum and maximum values of the training set that we used to normalize the data.
    minimum = -6.345355987548828
    maximum = 6.703166484832764

    output = (maximum - minimum) * (output + 1/2) + minimum

    return output

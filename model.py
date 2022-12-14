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
from parameters.nice import NICE
from parameters.nice.utils import Trainer

# <!> DO NOT ADD ANY OTHER ARGUMENTS <!>


def generative_model(noise):
    """
    Generative model

    Parameters
    ----------
    noise : ndarray with shape (n_samples, n_dim)
        input noise of the generative model
    """
    # Input dimension
    len_dim = 10
    # Use the first 10 dimensions of the noise in this example
    latent_variable = noise[:, :len_dim]
    # Load the model and the parameters
    model = NICE(prior=torch.distributions.Normal(torch.tensor(0.), torch.tensor(1.)),
                 coupling=4,
                 len_input=len_dim,
                 mid_dim=10,
                 hidden=4,
                 mask_config=1.)
    # Use the trainer class to load the parameters
    trainer = Trainer(model)
    model_trained = trainer.load_model("parameters/nice/model_1.pt")
    model_trained.eval()
    # Convert the numpy array to a torch tensor
    latent_variable = torch.from_numpy(latent_variable).float()
    # Generate the output
    output = model_trained.g(latent_variable).cpu().detach().numpy()
    return output

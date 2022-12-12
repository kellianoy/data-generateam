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
from models.nice import NICE
from models.nice.utils import Trainer

# <!> DO NOT ADD ANY OTHER ARGUMENTS <!>


def generative_model(noise):
    """
    Generative model

    Parameters
    ----------
    noise : ndarray with shape (n_samples, n_dim)
        input noise of the generative model
    """
    # See below an example
    # ---------------------
    len_dim = 10
    # use the first 10 dimensions of the noise in this example
    latent_variable = noise[:, :len_dim]
    # load my parameters (of dimension 15 in this example).
    # <!> be sure that they are stored in the parameters/ directory <!>
    model = NICE(prior=torch.distributions.Normal(torch.tensor(0.), torch.tensor(1.)),
                 coupling=4,
                 len_input=len_dim,
                 mid_dim=10,
                 hidden=4,
                 mask_config=1.)
    trainer = Trainer(model)
    model_trained = trainer.load_model("parameters/nice/model_1.pt")
    # model_trained = trainer.load_model("./parameters/nice/model_1.pt")
    model_trained.eval()
    # generate the output of the generative model
    output = latent_variable.copy()
    for i in range(latent_variable.shape[0]):
        for j in range(latent_variable.shape[1]):
            output[i][j] = np.mean(model_trained.g(
                latent_variable[i][j]).cpu().detach().numpy())
    # return the output of the generative model
    return output

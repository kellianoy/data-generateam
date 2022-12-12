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
    latent_variable = noise[:, :15]  # use the first 10 dimensions of the noise in this example

    # load my parameters (of dimension 15 in this example). 
    # <!> be sure that they are stored in the parameters/ directory <!>
    model = NICE(prior=latent_variable, 
            coupling=4, 
            len_input=15, 
            mid_dim=15, 
            hidden=4, 
            mask_config=1)
    model.load_state_dict(torch.load("./parameters/nice/model_1.pt"))
    model.eval()
    # generate the output of the generative model
    output = model(latent_variable)
    
    # return the output of the generative model
    return output 




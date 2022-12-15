import argparse
from data.dataset_tools import generate_basic_timeseries_splitted_normalized_dataset, denormalize_temperature
from data.dataset_pytorch import Dataset
import torch
from torch.utils.data import DataLoader
from tqdm import trange
from scipy.stats import anderson_ksamp
import copy
import os

import matplotlib.pyplot as plt
import numpy as np


def compute_anderson_darling( generated_sample, real_sample):
    n_station = real_sample.shape[1]
    anderson_darling = []
    for station in range(n_station):
        anderson_darling.append( anderson_ksamp([generated_sample[:,station], real_sample[:,station]])[0] )
    anderson_darling = np.array(anderson_darling)
    return np.mean(anderson_darling)

# def compute_anderson_darling(predictions, data):
#     N,P = data.shape
#     ADdistance = 0
#     for station in range(P) :
#         temp_predictions = predictions[:,station].reshape(-1)
#         temp_data = data[:,station].reshape(-1)
#         sorted_array = np.sort(temp_predictions)
#         count = np.zeros(len(temp_data))
#         count = (1/(N+2)) * np.array([(temp_data < order).sum()+1 for order in sorted_array])
#         idx = np.arange(1, N+1)
#         ADdistance = (2*idx - 1)* (np.log(count) + np.log(1-count[::-1]))
#         ADdistance = - N - np.sum(ADdistance)/N
#     return ADdistance/P

def compute_absolute_kendall_error( generated_sample, real_sample, n_test):
    R = []
    R_hat = []
    for i in range(n_test):
        R_i = 0
        R_i_hat = 0
        for j in range(n_test):
            if ( real_sample[j] < real_sample[i]).all():
                R_i += 1
            if ( generated_sample[j] < generated_sample[i]).all():
                R_i_hat += 1
        R.append(R_i/(n_test-1))
        R_hat.append(R_i_hat/(n_test-1))
    R = np.sort( np.array(R.sort()), axis = 0 ) 
    R_hat = np.sort( np.array(R_hat.sort()), axis = 0 )

    return np.linalg.norm((R-R_hat), ord=1)

def compute_error_on_test( temperature_test, time, trainer):

    n_test = time.shape[0]
    time_interval = [time[0],time[-1]]
    generated_sample = trainer.generate_sample(n_test , time_interval)

    anderson_darling_metrics = compute_anderson_darling( generated_sample, temperature_test)
    return anderson_darling_metrics



def main(args):

    dataset_name = args.dataset_name
    proportion_test = args.proportion_test
    model_type = args.model_type
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    lr = args.lr
    model_name = args.model_name

    dataset = generate_basic_timeseries_splitted_normalized_dataset(dataset_name, proportion_test = proportion_test)
    training_set = dataset[0][0]
    testing_set = dataset[0][1]
    max_temperature = dataset[1]
    min_temperature = dataset[2]

    model_type = args.model_type

    if model_type == "simple_gan":
        from parameters.simple_gan import Model
        from parameters.simple_gan import Trainer

        model = Model(len_input = 10)
        trainer = Trainer( model, lr)
        trainer.weights_init_uniform_rule(model)
        model_path = "parameters/simple_gan/models_saved/"
        model_name = "{}.pt".format(model_name)

    elif model_type == "nice":
        from parameters.nice import NICE
        from parameters.nice import Trainer
        sigma = 0.1
        noise_input = torch.distributions.Normal(
            torch.tensor(0.), torch.tensor(1.))
        coupling = 4
        len_input_output = 10
        mid_dim = 10
        hidden = 4
        mask_config = 1
        model = NICE(prior=noise_input, 
                coupling=coupling, 
                len_input=len_input_output, 
                mid_dim=mid_dim, 
                hidden=hidden, 
                mask_config=mask_config)
        trainer = Trainer( model, lr)

        model_path = "parameters/nice/models_saved/"
        model_name = "{}.pt".format(model_name)

    elif model_type == "nice_conditional":
        from parameters.nice_conditional import NICE_CONDITIONAL
        from parameters.nice_conditional import Trainer
        sigma = 0.1
        noise_input = torch.distributions.Normal(
            torch.tensor(0.), torch.tensor(1.))
        coupling = 4
        len_input_output = 10
        mid_dim = 10
        # hidden = 4
        hidden = 4
        mask_config = 1
        model = NICE_CONDITIONAL(prior=noise_input, 
                coupling=coupling, 
                len_input=len_input_output, 
                mid_dim=mid_dim, 
                hidden=hidden, 
                mask_config=mask_config)
        trainer = Trainer( model, lr)

        model_path = "parameters/nice_conditional/models_saved/"
        model_name = "{}.pt".format(model_name)

    else:
        raise NotImplementedError
        
    print("ok")
    print("")
    print("Data preparation...")
    temperature_training_set = torch.from_numpy(training_set[0]).float()
    time_training_set = torch.from_numpy(training_set[1]).float()
    temperature_testing_set = testing_set[0]
    time_testing_set = testing_set[1]

    torch_training = Dataset(temperature_training_set,time_training_set)

    train_loader = DataLoader(torch_training, batch_size=batch_size,
                                    shuffle=True, num_workers=0)
    testing_error = []
    error_on_train_set= []

    print("Training...")
    print("")

    model_trained = []

    for epoch in (pbar := trange(num_epochs)):
        for temperature ,time in train_loader:
            trainer.training_iteration(temperature, time)

        testing_error.append( compute_error_on_test(temperature_testing_set, time_testing_set, trainer) )
        error_on_train_set.append( compute_error_on_test(training_set[0], training_set[1], trainer) )

        model_trained.append(copy.deepcopy( trainer.model_to_save() ))

        pbar.set_description(f"Error on testing set: {testing_error[-1]}, on training set: {error_on_train_set[-1]}")

    testing_error = np.array(testing_error)

    best_metrics = np.min(testing_error)
    optimal_epoch = np.argmin(testing_error)

    print("")
    print("End of the training phase, minimum error reached : L = {} at epoch {}".format(best_metrics,optimal_epoch))
    print("")

    optimal_model = model_trained[optimal_epoch]

    isExist = os.path.exists(model_path)
    if not isExist:
        os.makedirs(model_path)

    torch.save({'model_state_dict':optimal_model.state_dict()}, model_path+model_name)

    print("Model saved at : {}".format(model_path))
    print("")

    plt.plot(error_on_train_set)
    plt.plot(testing_error)
    plt.legend(["Error on training set", "Error on testing set"])
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generating temperature models')

    parser.add_argument('--dataset_name', "--string", default="df_train" ,type=str)
    parser.add_argument('--proportion_test', default=0.8, type=float, help='peroportion testin dataset')
    parser.add_argument('--batch_size', default=64, type=int, help='Batch size')
    parser.add_argument('--lr', default=1e-3, type=float, help='Learning rate')
    parser.add_argument('--num_epochs', default=2, type=int, help='Number of epochs to train')
    parser.add_argument('--resume', '-r', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--model_type', default="nice_conditional" ,type=str)
    parser.add_argument('--model_name', default="model_1" ,type=str)

    main(parser.parse_args())
from metrics import Metrics
import argparse
from data.dataset_tools import generate_basic_timeseries_splitted_normalized_dataset_with_month_info, denormalize_temperature
from data.dataset_pytorch import Dataset, Timeseries_dataset
import torch
from torch.utils.data import DataLoader
from tqdm import trange
import copy
import os
import matplotlib.pyplot as plt
import numpy as np


def main(args):
    # Parse arguments
    dataset_name = args.dataset_name
    proportion_test = args.proportion_test
    model_type = args.model_type
    model_name = args.model_name
    model_loss = args.model_loss
    num_epochs = args.num_epochs
    batch_size = args.batch_size
    lr = args.lr

    # Load dataset and generat training and testing set (normalized)
    dataset = generate_basic_timeseries_splitted_normalized_dataset_with_month_info(
        dataset_name, proportion_test=proportion_test)
    training_set = dataset[0][0]
    testing_set = dataset[0][1]
    max_temperature = dataset[1]
    min_temperature = dataset[2]



    from parameters.nice_conditional import NICE_CONDITIONAL
    from parameters.nice_conditional import Trainer



    print("")
    print("Data preparation...")

    model_path = "parameters/{}/models_saved/".format(model_type)
    model_name = model_name + ".pt"
    # Preparating the data: dividing in training and testing sets
    temperature_training_set = torch.from_numpy(training_set[0]).float()
    time_training_set = torch.from_numpy(training_set[1]).float()
    temperature_testing_set = testing_set[0]
    time_testing_set = testing_set[1]
    
    torch_training = Dataset(temperature_training_set, time_training_set)

    train_loader = DataLoader(torch_training, batch_size=batch_size,
                              shuffle=True, num_workers=0)

    print("Training...")
    print("")

    # Metrics initialization

    ############IMPORTANT##############
    num_epochs = 300
    lr = 5e-4
    N_hp = 10
    windows_hp = range(0,N_hp)
    X = np.arange(0,N_hp)
    mean = np.zeros(N_hp)
    std = np.zeros(N_hp)

    n_of_model = range(0,10)
    #####################
    # Training loop
    for i in windows_hp:
        for j in n_of_model:

            noise_input = torch.distributions.Normal(
                torch.tensor(0.), torch.tensor(1.))
            coupling = 4
            len_input_output = 10
            mid_dim = 10
            hidden = 4
            mask_config = 1
            mid_time_dim = 7
            #####
            number_hidden_block_time = i
            #####
            model = NICE_CONDITIONAL(prior=noise_input,
                                    coupling=coupling,
                                    len_input=len_input_output,
                                    mid_dim=mid_dim,
                                    hidden=hidden,
                                    mask_config=mask_config,
                                    time_dim=13,
                                    mid_time_dim = mid_time_dim,
                                    number_hidden_block_time = number_hidden_block_time)
            trainer = Trainer(model, lr)

            time_series = False
            testing_error = []
            metrics = Metrics(trainer, model_loss)

            for epoch in range(num_epochs):
                for temperature, time in train_loader:
                    trainer.training_iteration(temperature, time)

                testing_error.append(metrics.compute_error_on_test(
                    temperature_testing_set, time_testing_set, time_series))
                    # error_on_train_set.append(metrics.compute_error_on_test(
                    #     training_set[0], training_set[1], time_series))

        mean[i] = np.mean(np.array(testing_error))
        std[i] =  np.std(np.array(testing_error))


    print("min is :", np.min(mean), 'and better hyper parameter: ', np.argmin(mean))
    plt.errorbar(X, mean, std, linestyle='None', marker='^')

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Generating temperature models')

    parser.add_argument('--dataset_name', "--string",
                        default="df_train", type=str)
    parser.add_argument('--proportion_test', default=0.8,
                        type=float, help='proportion test in dataset')
    parser.add_argument('--batch_size', default=64,
                        type=int, help='Batch size')
    parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
    parser.add_argument('--num_epochs', default=100,
                        type=int, help='Number of epochs to train')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='Resume from checkpoint')
    parser.add_argument('--model_type', default="nice_conditional", type=str)
    parser.add_argument('--model_name', default="model_1", type=str)
    parser.add_argument('--model_loss', default="ad", type=str)
    parser.add_argument('--memory_size', default=50, type=int)
    parser.add_argument('--number_ts', default=1, type=int)

    main(parser.parse_args())
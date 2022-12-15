import argparse
from data.dataset_tools import generate_basic_timeseries_splitted_normalized_dataset, denormalize_temperature
from data.dataset_pytorch import Dataset
import torch
from torch.utils.data import DataLoader
from tqdm import trange
import copy
import os
import matplotlib.pyplot as plt
import numpy as np
from metrics import Metrics


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

    # Load dataset and generate training and testing set (normalized)
    dataset = generate_basic_timeseries_splitted_normalized_dataset(
        dataset_name, proportion_test=proportion_test)
    training_set = dataset[0][0]
    testing_set = dataset[0][1]
    max_temperature = dataset[1]
    min_temperature = dataset[2]

    # Selecting the model
    if model_type == "simple_gan":
        from parameters.simple_gan import Model
        from parameters.simple_gan import Trainer

        model = Model(len_input=10)
        trainer = Trainer(model, lr)
        trainer.weights_init_uniform_rule(model)
        model_path = "parameters/simple_gan/"
        model_name = "{}.pt".format(model_name)

    if model_type == "nice":
        from parameters.nice import NICE
        from parameters.nice import Trainer
        noise_input = torch.distributions.Normal(
            torch.tensor(0.), torch.tensor(1.))
        coupling = 4
        len_input_output = 10
        mid_dim = 10
        hidden = 4
        mask_config = 1.
        model = NICE(prior=noise_input,
                     coupling=coupling,
                     len_input=len_input_output,
                     mid_dim=mid_dim,
                     hidden=hidden,
                     mask_config=mask_config)
        trainer = Trainer(model, lr)
        model_path = "parameters/nice/"
        model_name = "{}.pt".format(model_name)

    else:
        raise NotImplementedError

    print("")
    print("Data preparation...")

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
    model_trained = []
    testing_error = []
    metrics = Metrics(trainer)

    # Training loop
    for epoch in (pbar := trange(num_epochs)):
        for temperature, time in train_loader:
            trainer.training_iteration(temperature, time)

        testing_error.append(metrics.compute_error_on_test(
            temperature_testing_set, time_testing_set, mode=model_loss))

        model_trained.append(copy.deepcopy(trainer.model_to_save()))

        pbar.set_description(f"Error on testing set: {testing_error[-1]}")

    # Collecting the best model
    testing_error = np.array(testing_error)
    best_metrics = np.min(testing_error)
    optimal_epoch = np.argmin(testing_error)

    print("")
    print("End of the training phase, minimum error reached : L = {} at epoch {}".format(
        best_metrics, optimal_epoch))
    print("")

    # Saving the model
    optimal_model = model_trained[optimal_epoch]

    isExist = os.path.exists(model_path)
    if not isExist:
        os.makedirs(model_path)

    torch.save({'model_state_dict': optimal_model.state_dict()},
               model_path+model_name)

    print("Model saved at : {}".format(model_path))
    print("")

    # Plotting the error
    plt.plot(testing_error)
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
    parser.add_argument('--num_epochs', default=1000,
                        type=int, help='Number of epochs to train')
    parser.add_argument('--resume', '-r', action='store_true',
                        help='Resume from checkpoint')
    parser.add_argument('--model_type', default="nice", type=str)
    parser.add_argument('--model_name', default="model_1", type=str)
    parser.add_argument('--model_loss', default="ad", type=str)

    main(parser.parse_args())

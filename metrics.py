from scipy.stats import anderson_ksamp
import numpy as np


class Metrics:
    # Class constructor
    def __init__(self, trainer, mode):
        # @param trainer: the trainer object
        # @param mode: the metric to use

        self.trainer = trainer
        self.mode = mode

    # Method to compute anderson darling error
    def anderson_darling(self, generated_sample, real_sample):
        # Compute the anderson darling error
        # @param generated_sample: the generated sample
        # @param real_sample: the real sample

        n_station = real_sample.shape[1]
        anderson_darling = np.zeros(n_station)
        for station in range(n_station):
            anderson_darling[station] = anderson_ksamp(
                [generated_sample[:, station], real_sample[:, station]])[0]
        return np.mean(anderson_darling)

    def absolute_kendall_error(self, generated_sample, real_sample):
        # Compute the kendall's dependance function
        # @param generated_sample: the generated sample
        # @param real_sample: the real sample

        n_test = real_sample.shape[0]
        # COmpute the ones of the real sample
        R = (1/(n_test-1))*np.sum(real_sample[:, None] < real_sample, axis=1)
        # Compute the ones of the generated sample
        R_tild = (1/(n_test-1)) * \
            np.sum(generated_sample[:, None] < generated_sample, axis=1)
        # Return the linear norm of the difference (absolute error)
        return np.linalg.norm((R-R_tild), ord=1)

    def compute_error_on_test(self, temperature_test, time):
        # Compute the error on the test set using the chosen metric
        # @param temperature_test: the test set
        # @param time: the time vector
        # @param mode: the metric to use

        n_test = time.shape[0]
        time_interval = [time[0], time[-1]]
        generated_sample = self.trainer.generate_sample(n_test, time_interval)
        metric = None

        if self.mode == "ad":
            metric = self.anderson_darling(
                generated_sample, temperature_test)

        elif self.mode == "ke":
            metric = self.absolute_kendall_error(
                generated_sample, temperature_test)

        else:
            raise NotImplementedError

        return metric

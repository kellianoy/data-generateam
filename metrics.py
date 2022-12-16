from scipy.stats import anderson_ksamp
import numpy as np
# high-speed compiler
from numba import njit

# Redefining np.all(x, axis=1) for numba
@njit(cache=True)
def np_all_axis1(x):
    """Numba compatible version of np.all(x, axis=1)."""
    out = np.ones(x.shape[0], dtype=np.bool8)
    for i in range(x.shape[1]):
        out = np.logical_and(out, x[:, i])
    return out

@njit(cache=True)
def ake(generated_samples, real_samples):
    # Compute the kendall's dependance function
    # @param generated_samples: the generated sample
    # @param real_samples: the real sample
    n_test = real_samples.shape[0]
    R_i = np.zeros((n_test, n_test))
    R_i_tild = np.zeros((n_test, n_test))
    for j in range(n_test):
        # R_i is the sum of the rows that are smaller than the ith row
        R_i[j] = np_all_axis1(real_samples[j] < real_samples)
        R_i_tild[j] = np_all_axis1(generated_samples[j] < generated_samples)
    R = np.sort(1/(n_test-1) * np.sum(R_i, axis=0))
    R_tild = np.sort(1/(n_test-1)* np.sum(R_i_tild, axis=0))
    return np.linalg.norm((R-R_tild), ord=1)

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

    def absolute_kendall_error(self, generated_samples, real_samples):
        # Compute the kendall's dependance function
        # @param generated_samples: the generated sample
        # @param real_samples: the real sample
        return ake(generated_samples, real_samples)

    def compute_error_on_test(self, temperature_test, time, time_series, past_infos = None,number_ts = None):
        # Compute the error on the test set using the chosen metric
        # @param temperature: the test set
        # @param time: the time vector
        # @param mode: the metric to use

        n_test = time.shape[0]
        time_interval = [time[0], time[-1]]
        if time_series:
            generated_sample = self.trainer.generate_sample(n_test, temperature_test[0], time, past_infos ,number_ts)
        else:
            generated_sample = self.trainer.generate_sample(n_test, time_interval)
        metric = None

        if self.mode == "ad":
            metric = self.anderson_darling(
                generated_sample, temperature)

        elif self.mode == "ke":
            metric = self.absolute_kendall_error(
                generated_sample, temperature)

        else:
            raise NotImplementedError

        return metric

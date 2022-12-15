from scipy.stats import anderson_ksamp
import numpy as np

class Metrics:
    # Class constructor
    def __init__(self, trainer):
        self.trainer = trainer
        
    # Method to compute anderson darling error
    def compute_anderson_darling(self, generated_sample, real_sample):
        n_station = real_sample.shape[1]
        anderson_darling = []
        for station in range(n_station):
            anderson_darling.append(anderson_ksamp(
                [generated_sample[:, station], real_sample[:, station]])[0])
        anderson_darling = np.array(anderson_darling)
        return np.mean(anderson_darling)

    def compute_absolute_kendall_error(self, generated_sample, real_sample, n_test):
        R = []
        R_hat = []
        for i in range(n_test):
            R_i = 0
            R_i_hat = 0
            for j in range(n_test):
                if (real_sample[j] < real_sample[i]).all():
                    R_i += 1
                if (generated_sample[j] < generated_sample[i]).all():
                    R_i_hat += 1
            R.append(R_i/(n_test-1))
            R_hat.append(R_i_hat/(n_test-1))
        R = np.sort(np.array(R.sort()), axis=0)
        R_hat = np.sort(np.array(R_hat.sort()), axis=0)

        return np.linalg.norm((R-R_hat), ord=1)

    def compute_error_on_test(self, temperature_test, time, mode="ad"):
        # Compute the error on the test set using the chosen metric
        # @param temperature_test: the test set
        # @param time: the time vector
        # @param mode: the metric to use
        n_test = time.shape[0]
        time_interval = [time[0], time[-1]]
        generated_sample = self.trainer.generate_sample(n_test, time_interval)
        metric = None
        
        if mode == "ad":
            metric = self.compute_anderson_darling(
                generated_sample, temperature_test)
            
        elif mode == "ke":
            metric = self.compute_absolute_kendall_error(generated_sample, temperature_test, n_test)
        
        else:
            raise NotImplementedError
        
        return metric

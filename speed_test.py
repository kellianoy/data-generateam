import numpy as np
from time import sleep
import timeit

def kellian_ke(generated_samples, real_samples):
    # Compute the kendall's dependance function
    # @param generated_samples: the generated sample
    # @param real_samples: the real sample
    n_test = real_samples.shape[0]
    R_i = np.zeros((n_test, n_test))
    R_i_tild = np.zeros((n_test, n_test))
    for j in range(n_test):
        # R_i is the sum of the rows that are smaller than the ith row
        R_i[j] = (real_samples[j] < real_samples).all(axis=1)
        R_i_tild[j] = (generated_samples[j] < generated_samples).all(axis=1)
    R = np.sort(1/(n_test-1) * np.sum(R_i, axis=0))
    R_tild = np.sort(1/(n_test-1)* np.sum(R_i_tild, axis=0))
    return np.linalg.norm((R-R_tild), ord=1)

def thomas_ke(generated_sample, real_sample):
    n_test = real_sample.shape[0]
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
    R = np.sort( np.asarray(R), axis = 0 ) 
    R_hat = np.sort(np.asarray(R_hat), axis=0) 

    return np.linalg.norm((R-R_hat), ord=1)

if __name__ == "__main__":
    # generate random samples between 0 and 1 of size 1924 x 10
    generated_sample = np.random.rand(1000, 10)
    real_sample = np.random.rand(1000, 10)
    # time it
    assert kellian_ke(generated_sample, real_sample) == thomas_ke(generated_sample, real_sample)
    print("NumPy KE: ", timeit.timeit(lambda: kellian_ke(generated_sample, real_sample), number=20))
    print("Python KE: ", timeit.timeit(lambda: thomas_ke(generated_sample, real_sample), number=20))

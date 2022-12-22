import numpy as np
from time import sleep
import timeit
from numba import njit
from math import isclose

def python_ke(generated_sample, real_sample):
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

def numpy_ke(generated_samples, real_samples):
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

@njit(cache=True)
def np_all_axis1(x):
    """Numba compatible version of np.all(x, axis=1)."""
    out = np.ones(x.shape[0], dtype=np.bool8)
    for i in range(x.shape[1]):
        out = np.logical_and(out, x[:, i])
    return out

@njit(cache=True)
def numba_ke(generated_samples, real_samples):
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

if __name__ == "__main__":
    # generate random samples between 0 and 1 of size 1924 x 10
    samples = 1924
    stations = 10

    generated_samples = np.random.rand(samples, stations)
    real_samples = np.random.rand(samples, stations)
    # Python KE is very long to compute, don't exceed 20 iterations (~ 1,2 seconds per iteration)
    n_iter = 50
    epsilon = 1e-6
    
    # Assertions to check that the results are the same
    # assert isclose(python_ke(generated_samples, real_samples), numpy_ke(generated_samples, real_samples), rel_tol=epsilon, abs_tol=0.0)
    assert isclose(numba_ke(generated_samples, real_samples), numpy_ke(generated_samples, real_samples), rel_tol=epsilon, abs_tol=0.0)
    
    # time it
    # print("Python KE per iteration: ", timeit.timeit(lambda: python_ke(generated_samples, real_samples), number=n_iter) / n_iter)
    print("NumPy KE per iteration: ", timeit.timeit(lambda: numpy_ke(generated_samples, real_samples), number=n_iter) / n_iter)
    print("NumPy Boost KE per iteration: ", timeit.timeit(lambda: numba_ke(generated_samples, real_samples), number=n_iter) / n_iter)

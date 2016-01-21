
from __future__ import division, print_function
import numpy as np
from scipy.signal import fftconvolve

def noise_white(n, rows=1):

    return np.random.randn(rows, n)

def noise_pink(n, rows=1, alpha=0.1):

    X = np.fft.rfft(np.random.randn(rows, n), axis=1)
    X[:,1:] /= np.arange(1,X.shape[1])/n
    X[:,0] = 0.
    x = np.fft.irfft(X, axis=1)
    x = (x.T/np.sqrt(np.mean(x**2, axis=1))).T

    return x

def noise_ar1(n, rows=1, a1=0.9):

    x = noise_white(n, rows=rows)
    for row in x:
        row[:] = fftconvolve(row, np.array([1., a1]), mode='same')
    x = (x.T/np.sqrt(np.mean(x**2, axis=1))).T

    return x

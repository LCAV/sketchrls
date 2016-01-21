
from __future__ import division, print_function
import numpy as np
from scipy import fftpack

def autocorr(x):
    """ Fast autocorrelation computation using the FFT """
    
    X = np.fft.rfft(x, n=2*x.shape[0])
    r = np.fft.irfft(X*np.conj(X))
    
    return r[:x.shape[0]]/x.shape[0]

def toeplitz_multiplication(c, r, A):
    """ Compute numpy.dot(scipy.linalg.toeplitz(c,r), A) using the FFT. """
    
    m = c.shape[0]
    n = r.shape[0]

    fft_len = 2**np.ceil(np.log2(m+n-1))
    
    if A.shape[0] != n:
        raise ValueError('A dimensions not compatible with toeplitz(c,r)')
    
    x = np.concatenate((c, r[-1:0:-1]))
    xf = np.fft.rfft(x, n=fft_len)
    
    Af = np.fft.rfft(A, n=fft_len, axis=0)
    
    return np.fft.irfft((Af.T*xf).T, n=fft_len, axis=0)[:m,]

def hankel_multiplication(c, r, A):
    """ Compute numpy.dot(scipy.linalg.hankel(c,r=r), A) using the FFT. """
    
    return toeplitz_multiplication(c[::-1], r, A)[::-1,]

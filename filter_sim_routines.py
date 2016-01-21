from __future__ import division, print_function
import numpy as np
import time
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt

import pyroomacoustics as pra

from noise import *

def generate_signal(n, p, loops, SNR_dB=100, noise='white', h=None):

    # First generate a random signal
    if noise == 'pink':
        x = noise_pink(n, rows=loops, alpha=1e-10)
    elif noise == 'ar1':
        x = noise_ar1(n, rows=loops)
    else:
        x = noise_white(n, rows=loops)

    # Generate random filters on the sphere
    if h is None:
        h = np.random.randn(loops,p)
        norm = np.linalg.norm(h, axis=1)
        h = (h.T/norm).T
    
    if h.ndim == 1:
        if h.shape[0] >= p:
            h = np.tile(h[:p], (loops,1))
        else:
            h2 = np.zeros(loops,p)
            for i in xrange(loops):
                h2[i,:h.shape[0]] = h
            h = h2

    # Finally generate the filtered signal
    sigma_noise = 10.**(-SNR_dB/20.)
    d = np.zeros((loops,n+h.shape[1]-1))
    for l in xrange(loops):
        d[l,:] = fftconvolve(x[l], h[l])
        d[l,:] += np.random.randn(n+h.shape[1]-1)*sigma_noise

    return x, h, d


def shoebox_rir(room_dim, source, mic):

    # Some simulation parameters
    Fs = 8000
    t0 = 1./(Fs*np.pi*1e-2)  # starting time function of sinc decay in RIR response
    absorption = 0.90
    max_order_sim = 10

    # create a microphone array
    R = pra.linear2DArray(mic, 1, 0, 1) 
    mics = pra.Beamformer(R, Fs)

    # create the room with sources and mics
    room1 = pra.Room.shoeBox2D(
        [0,0],
        room_dim,
        Fs,
        t0 = t0,
        max_order=max_order_sim,
        absorption=absorption,
        sigma2_awgn=0)

    # add source and interferer
    room1.addSource(source)
    room1.addMicrophoneArray(mics)

    room1.compute_RIR()
    h = room1.rir[0][0]

    return h



def run_filter(x, d, fil):
    
    w = np.zeros((x.shape[0], fil.length))
    for i in xrange(x.shape[0]):
        fil.update(x[i], d[i])
        w[i,:] = fil.w[:]
        
    return w


def test_adaptive_filter(x, d, fil, h, rng_seed=0, loops=1):
    """ Run the adaptive filter on data and plot output """

    # fix randomness
    np.random.seed(rng_seed)

    e = np.zeros(x.shape[1])
    ellapsed = 0.

    if x.ndim == 1:
        x = np.array([x])
    elif x.ndim > 2:
        raise ValueError('Too many dimensions')

    for l in xrange(x.shape[0]):

        fil.reset()

        start = time.time()

        w = run_filter(x[l], d[l], fil)

        end = time.time()
        ellapsed += end - start

        e += np.linalg.norm(h[l] - w, axis=1)**2

    M = np.minimum(3, h.shape[1])
    print(fil.name(),fil.w[:M],'time:',ellapsed/loops,'error:',np.linalg.norm(fil.w-h[-1])**2)

    plt.semilogy(e/loops)
    plt.ylim((0, 1.05))

    return e


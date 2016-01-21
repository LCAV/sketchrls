from __future__ import division, print_function
import numpy as np
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt

from adaptive_filters import *
from filter_sim_routines import *
from sketch_rls import *

n = 40000
p = 1000

SNR_dB = [10,20,30]

# RLS params
lmbd = 0.9999
delta = [10, 8, 6, 4]

# NLMS params
step = 1.

# SketchRLS params
N = 5
pr = 0.005

# Monte-Carlo simulation parameters
loops = 1

# create all the filters
filters = []
for i in xrange(len(SNR_dB)):
    nlms = NLMS(p, mu=step)
    rls = RLS(p, lmbd=lmbd, delta=delta[i])
    brls = BlockRLS(p, lmbd=lmbd, delta=delta[i], L=int(1/pr))
    srls = SketchRLS(p, lmbd=lmbd, delta=delta[i], N=N, p=pr)

    filters.append([nlms, brls, srls])

# start RNG
rng_seed = int(100000*(time.time() % 1))
print('RNG Seed:',rng_seed)

plt.figure()

wsp = 2
hsp = 2

for i in xrange(len(SNR_dB)):

    plt.subplot(hsp, wsp, i+1)

    # draw random samples
    x, h, d = generate_signal(n, p, loops, SNR_dB[i], noise='ar1')

    # run the filters
    for fil in filters[i]:
        test_adaptive_filter(x, d, fil, h, rng_seed=rng_seed, loops=loops)

    labels = [f.name() for f in filters[i]]

    plt.legend(labels)
    plt.title(SNR_dB[i])

plt.show()


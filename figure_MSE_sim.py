from __future__ import division, print_function
import numpy as np
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt

from adaptive_filters import *
from filter_sim_routines import *
from sketch_rls import *

def mse_nlms(x, d, h, mu):
    import numpy as np
    from adaptive_filters import NLMS
    from filter_sim_routines import run_filter
    nlms = NLMS(h.shape[0], mu=mu)
    w = run_filter(x, d, nlms)
    e = np.linalg.norm(h - w, axis=1)**2
    return e

def mse_brls(x, d, h, lmbd, delta, L):
    import numpy as np
    from adaptive_filters import BlockRLS
    from filter_sim_routines import run_filter
    brls = BlockRLS(h.shape[0], lmbd=lmbd, delta=delta, L=L)
    w = run_filter(x, d, brls)
    e = np.linalg.norm(h - w, axis=1)**2
    return e

def mse_srls(x, d, h, lmbd, delta, N, pr):
    import numpy as np
    from sketch_rls import SketchRLS
    from filter_sim_routines import run_filter
    srls = SketchRLS(h.shape[0], lmbd=lmbd, delta=delta, N=N, p=pr)
    w = run_filter(x, d, srls)
    e = np.linalg.norm(h - w, axis=1)**2
    return e

if __name__ == '__main__':

    filters = ['NLMS', 'BlockRLS', 'RHS']

    # Monte-Carlo simulation parameters
    loops = 1

    # PARAMETERS TO SWEEP
    #####################
    SNR_dB = [10,20,30]
    noises = ['white','ar1']

    # signal length and filter dimension
    n = {'white':20000, 'ar1':40000}
    p = 1000

    # RLS params
    # forgetting factor
    lmbd = [0.9999]
    ff = np.ones(loops)*lmbd
    # regularization parameter
    delta = [20, 10, 10, 10]

    # NLMS params
    step = [0.1, 0.2, 0.5, 1.]

    # BlockRLS params
    L = {'white':[200], 'ar1':[200]}

    # SketchRLS params
    N  =  [20,    10,    5,    5,    5,     5,     2]
    pr =  [0.005, 0.005, 0.05, 0.01, 0.005, 0.001, 0.005]
    if len(N) != len(pr):
        raise ValueError('N and pr must have the same number of elements')

    # start timing simulation
    start = time.time()

    # Launch many workers!
    from IPython import parallel

    # setup parallel computation env
    c = parallel.Client()
    #c = parallel.Client('/Users/scheibler/.starcluster/ipcluster/SecurityGroup:@sc-mycluster-us-east-1.json', sshkey='/Users/scheibler/.ssh/aws_key.rsa')
    print(c.ids)
    c.blocks = True
    view = c.load_balanced_view()

    # NESTED SIMULATION LOOP
    ########################

    for noise in noises:

        # start RNG
        rng_seed = int(100000*(time.time() % 1))
        print('RNG Seed (' + noise + '):',rng_seed)

        # DATA RECEPTACLE
        #################
        e_nlms = np.zeros((len(SNR_dB), len(step), n[noise], loops))
        e_brls = np.zeros((len(SNR_dB), len(L[noise]), n[noise], loops))
        e_srls = np.zeros((len(SNR_dB), len(N), n[noise], loops))

        for i,snr in enumerate(SNR_dB):

            # INPUT DATA
            ############

            x, h, d = generate_signal(n[noise], p, loops, SNR_dB=snr, noise=noise)

            # SIMULATION
            ############

            print('LMS ' + noise + ' SNR ' + str(snr))
            for s,mu in enumerate(step):
                mu0 = np.ones(loops)*mu
                e_nlms[i,s,:,:] = np.array(view.map_sync(mse_nlms, x, d, h, mu0)).T

            print('RLS ' + noise + ' SNR ' + str(snr))
            dlta = np.ones(loops)*delta[i]
            for s,l0 in enumerate(L[noise]):
                l = np.ones(loops,dtype='int')*l0
                e_brls[i,s,:,:] = np.array(view.map_sync(mse_brls, x, d, h, ff, dlta, l)).T

            print('RHS ' + noise + ' SNR ' + str(snr))
            for s,(N0,p0) in enumerate(zip(N,pr)):
                pp = np.ones(loops)*p0
                NN = np.ones(loops, dtype='int')*N0
                e_srls[i,s,:,:] = np.array(view.map_sync(mse_srls, x, d, h, ff, dlta, NN, pp)).T

        filename = 'sim_data/MSE_' + noise + '_' + time.strftime('%Y%m%d-%H%M%S') + '.npz'
        np.savez(filename,
                n=n[noise], p=p, rng_seed=rng_seed,
                SNR_dB=SNR_dB, noise=noise, lmbd=lmbd, delta=delta, step=step, 
                L=L[noise], N=N, pr=pr,
                nlms=e_nlms, brls=e_brls, srls=e_srls)

    end = time.time()
    print('Total time ellapsed:',end-start)


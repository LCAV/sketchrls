
from __future__ import division, print_function
import numpy as np
from scipy.signal import fftconvolve
import matplotlib.pyplot as plt

label_font_size = 8
tick_font_size = 6

def set_color_map(ax1, length):
        newmap = plt.get_cmap('summer')
        ax1 = plt.gca()
        ax1.set_color_cycle([newmap( k ) for k in np.linspace(0.1,0.7,length)])


def set_axis(ax1, spi, title, color, snr, xb):
        plt.tick_params(
            axis='both',          # changes apply to the x-axis
            which='both',      # both major and minor ticks are affected
            bottom='off',      # ticks along the bottom edge are off
            top='off',         # ticks along the top edge are off
            right='off', left='off',
            labelbottom='off', labelleft='off') # labels along the bottom edge are off

        ax1.set_ylim(10**(-SNR_dB[i]/10-1), 1)

        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        ax1.spines['left'].set_visible(False)
        ax1.spines['bottom'].set_visible(False)

        v_axis = []
        axis_pos = {'bottom':5, 'left':7}
        if spi % 2 == 0:
            plt.ylabel('SNR ' + str(snr) + ' dB', fontsize=label_font_size)
            ax1.yaxis.set_label_position("right")
        if (spi-1) % 2 == 0:
            v_axis.append('left')
            ax1.yaxis.set_ticks_position('left')
            #plt.ylabel('MSE', fontsize=label_font_size)
        if spi > 4:
            v_axis.append('bottom')
            ax1.xaxis.set_ticks_position('bottom')
            ax1.xaxis.set_ticks([0,xb])
            ax1.xaxis.set_ticklabels(['0',str(xb//1000)+'k'])
            plt.xlabel(title, fontsize=label_font_size)
        for axis in v_axis:
            ax1.spines[axis].set_visible(True)
            ax1.spines[axis].set_position(('outward', axis_pos[axis]))
            ax1.spines[axis].set_linewidth(0.3)

        # Set ticks fontsize
        plt.xticks(size=tick_font_size)
        plt.yticks(size=tick_font_size)


noises = ['white','ar1']
lms_step_choice = {'ar1':[2,2,3], 'white':[2,2,2]}

sim_dir = 'sim_data'
# Add the date-time stamps of the simulation result files you want to plot to the array
# Make sure the date-time matches 'white' or 'ar1' in the name of the file.
files = { 'ar1':['20150926-010959',], # replace by what you need
        'white':['20150926-010428',], }

for n,noise in enumerate(noises):

    plt.figure(figsize=(8*0.394, 7.5*0.394))

    filename = sim_dir + '/MSE_' + noise + '_' + files[noise][0] + '.npz'

    e = np.load(filename)
    print(filename)
    e_nlms = e['nlms']
    e_brls = e['brls']
    e_srls = e['srls']

    SNR_dB = e['SNR_dB']
    lms_step = e['step']
    print('lms_step:',lms_step)
    N = np.array(e['N'])
    print('N:',N)
    pr = np.array(e['pr'])
    print('pr',pr)
    delta = e['delta']
    print('delta',delta)
    lmbd = e['lmbd']
    #n_bound = e['n']
    n_bound = e_nlms.shape[2]
    p = e['p']
    L = e['L']
    if len(L) > 1:
        raise ValueError('Plotting code does not support value of L larger than 1.')

    N_fixed = 5
    pr_fixed = 0.005
    I_N_fixed = (N == N_fixed)
    I_p_fixed = (pr == pr_fixed)

    for f in files[noise][1:]:
        filename = sim_dir + '/MSE_' + noise + '_' + f + '.npz'
        print(filename)
        e = np.load(filename)
        e_nlms = np.concatenate((e_nlms, e['nlms']), axis=-1)
        e_brls = np.concatenate((e_brls, e['brls']), axis=-1)
        e_srls = np.concatenate((e_srls, e['srls']), axis=-1)

    print('# repetitions:', e_nlms.shape[-1])

    for s,i in enumerate([0,1,2]):

        """ Now fixed N, variable pr """
        spi = s*2 + 1
        ax1 = plt.subplot(3,2,spi)

        set_color_map(ax1, np.sum(I_N_fixed))

        k = lms_step_choice[noise][s]
        plt.semilogy(e_nlms[i,k,:,:].mean(axis=-1).T, 'k:', linewidth=0.6, alpha=1., dashes=[2,2])
        plt.semilogy(np.arange(0, n_bound, L[0]), e_brls[i,0,::L[0],:].mean(axis=-1).T, 'k--', clip_on=False, linewidth=0.6, alpha=1., dashes=[4,2])
        plt.semilogy(e_srls[i,I_N_fixed,:,:].mean(axis=-1).T, '-', clip_on=False, linewidth=0.7, alpha=0.9)

        set_axis(ax1, spi, '$N='+str(N_fixed)+'$', noise, SNR_dB[i], n_bound)

        """ Now fixed pr, variable N """
        spi = s*2 + 2
        ax1 = plt.subplot(3,2,spi)

        set_color_map(ax1, np.sum(I_p_fixed))

        k = lms_step_choice[noise][s]
        plt.semilogy(e_nlms[i,k,:,:].mean(axis=-1).T, 'k:', linewidth=0.6, alpha=1., dashes=[2,2])
        plt.semilogy(np.arange(0, n_bound, L[0]), e_brls[i,0,::L[0],:].mean(axis=-1).T, 'k--', clip_on=False, linewidth=0.6, alpha=1, dashes=[4,2])
        plt.semilogy(e_srls[i,I_p_fixed,:,:].mean(axis=-1).T, '-', clip_on=False, linewidth=0.7, alpha=0.9)

        set_axis(ax1, spi, '$p='+str(pr_fixed)+'$', noise, SNR_dB[i], n_bound)
       
    plt.tight_layout(pad=0.1, w_pad=0.5, h_pad=0.1)
    plt.savefig('figure_MSE_' + noise + '.pdf')

plt.show()


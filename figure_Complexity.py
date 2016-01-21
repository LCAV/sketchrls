
from __future__ import division, print_function
import numpy as np
import matplotlib.pyplot as plt

def brls_complexity(d, L):
    return (d+L)*(d+L+1)*np.log(d+L)/L + d**2 + (2*d**2 + d)/L + (L+1)*d + L**2

def srls_complexity(d, p, N):
    L = 1/p
    q = 1 - np.exp(np.log(1-p)/N)
    return (d+1)*(d+L)*np.log(d+L)/L + 3*d/L + 3*N*q*(d**2+d) + N/L*(2*d**2+3*d)

if __name__ == '__main__':

    label_font_size = 8
    tick_font_size = 6

    plt.figure(figsize=(4*0.394, 4*0.394))
    cmap = plt.get_cmap('summer')

    col = np.linspace(0.1,0.7,4)
    dim = [1000, 100, 50, 10]
    pos = [(39.5, 0.005),
            (39.5, 0.01675),
            (39.5, 0.025),
            (39.5, 0.045),]

    for d,tpos,c in zip(dim, pos, col):

        p = np.linspace(1e-5,0.1,1000)
        N = np.arange(2,40.1,0.1)

        Nv, pv = np.meshgrid(N,p)

        brls_comp = brls_complexity(d, 1/pv)
        srls_comp = srls_complexity(d, pv, Nv)

        plt.imshow(c*(srls_comp >= brls_comp), cmap=cmap, interpolation="None", 
                    aspect='auto', extent=(N[0],N[-1],p[0],p[-1]), 
                    origin='lower', alpha=0.15)

        ax = plt.gca()
        ax.text(tpos[0], tpos[1], '$d = %d$'%(d),
            verticalalignment='center', horizontalalignment='right',
            color='black', fontsize=tick_font_size)

    ax1 = plt.gca()
    ax1.text(39, 0.09, 'RLS faster',
        verticalalignment='center', horizontalalignment='right',
        color='black', fontsize=tick_font_size)
    ax1.text(3.5, 0.01, 'RHS faster',
        verticalalignment='center', horizontalalignment='left',
        color='black', fontsize=tick_font_size)

    plt.tick_params(
        axis='both',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom='off',      # ticks along the bottom edge are off
        top='off',         # ticks along the top edge are off
        right='off', left='off',
        labelbottom='off', labelleft='off') # labels along the bottom edge are off

    ax1.xaxis.set_ticks_position('bottom')
    ax1.xaxis.set_ticks([int(N[0]),int(N[-1])])

    ax1.yaxis.set_ticks_position('left')
    ax1.yaxis.set_ticks([0,p[-1]])

    plt.ylabel('$p$', fontsize=label_font_size)
    plt.xlabel('$N$', fontsize=label_font_size)

    # Set ticks fontsize
    plt.xticks(size=tick_font_size)
    plt.yticks(size=tick_font_size)

    plt.tight_layout(pad=0.1)

    plt.savefig('figure_complexity.pdf')
    plt.show()

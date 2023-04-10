import numpy as np
import matplotlib.pyplot as plt
from spectrum import SpectrumSimple


def plot_spectrum(s: SpectrumSimple, R=True, T=False, ax=None, fig=None):
    wls = s.WLS
    assert R or T, "you have to plot something..."
    if R and T:
        if ax is None or fig is None:
            fig, ax = plt.subplots(1, 2)
        fig.tight_layout()

        R_arr = s.get_R()
        T_arr = s.get_T()
        ax[0].plot(wls, R_arr, label='R')
        ax[1].plot(wls, T_arr, label='T')

        ax[0].set_xlabel('wl / nm')
        ax[1].set_xlabel('wl / nm')
        ax[0].set_title('R')
        ax[1].set_title('T')

        ax[0].set_xlim(np.min(wls), np.max(wls))
        ax[1].set_xlim(np.min(wls), np.max(wls))
        ax[0].set_ylim(0., 1.)
        ax[1].set_ylim(0., 1.)

    else:
        if ax is None:
            fig, ax = plt.subplots(1, 1)
        if R:
            ax.plot(wls, s.get_R(), label='R')
        else:
            ax.plot(wls, s.get_T(), label='T')

        ax.set_xlabel('wl / nm')
        title = "R" if R else "T"
        ax.set_title(title)
        ax.set_xlim(np.min(wls), np.max(wls))
        ax.set_ylim(0., 1.)
    return ax

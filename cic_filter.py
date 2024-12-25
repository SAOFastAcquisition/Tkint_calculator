import numpy as np               # Import numpy
import matplotlib.pyplot as plt  # Import matplotlib
from scipy.signal import freqz
from scipy.fftpack import fft


class CicFilter:
    """
    Cascaded Integrator-Comb (CIC) filter is an optimized class of
    finite impulse response (FIR) filter.
    CIC filter combines an interpolator or decimator, so it has some
    parameters:

    R - decimation or interpolation ratio,
    N - number of stages in filter (or filter order)
    M - number of samples per stage (1 or 2)*

    * for this realisation of CIC filter just leave M = 1.

    CIC filter is used in multi-rate processing. In hardware
    applications CIC filter doesn't need multipliers, just only
    adders / subtractors and delay lines.

    Equation for 1st order CIC filter:
    y[n] = x[n] - x[n-RM] + y[n-1].


    Parameters
    ----------
    x : np.array
        input signal
    """

    def __init__(self, x):
        self.x = x

    def decimator(self, r, n):
        """
        CIC decimator: Integrator + Decimator + Comb

        Parameters
        ----------
        r : int
            decimation rate
        n : int
            filter order
        """

        # integrator
        y = self.x[:]
        for i in range(n):
            y = np.cumsum(y)

        # decimator

        y = y[::r]
        # comb stage
        return np.diff(y, n=n, prepend=np.zeros(n))

    def interpolator(self, r, n, mode=False):
        """
        CIC inteprolator: Comb + Decimator + Integrator

        Parameters
        ----------
        r : int
            interpolation rate
        n : int
            filter order
        mode : bool
            False - zero padding, True - value padding.
        """

        # comb stage
        y = np.diff(self.x, n=n,
                    prepend=np.zeros(n), append=np.zeros(n))

        # interpolation
        if mode:
            y = np.repeat(y, r)
        else:
            y = np.array([i if j == 0 else 0 for i in y for j in range(r)])

        # integrator
        for i in range(n):
            y = np.cumsum(y)

        if mode:
            return y[1:1 - n * r]
        else:
            return y[r - 1:-n * r + r - 1]


def plot_filter(r=None, n=None, samples=100, mode=None):
    # Create signal
    tt = np.linspace(0, 1, samples)

    np.random.seed(1)
    if mode == 'Decimator':
        x = 1.5 * np.sin(4 * np.pi * tt) + 1.7 * np.sin(8.3 * np.pi * tt)
        x += 0.9 * np.random.randn(samples)
    if mode == 'Interpolator':
        x = np.sin(1.7 * np.pi * tt) + 1.7 * np.sin(5.3 * np.pi * tt)
        x += 0.3 * np.random.randn(samples)

    # Apply filter
    clf = CicFilter(x)

    if mode == 'Decimator':
        zz = [clf.decimator(i, j) for i, j in zip(r, n)]
    if mode == 'Interpolator':
        zz = [clf.interpolator(i, j, mode=True) for i, j in zip(r, n)]

    # Plot figure
    plt.figure(figsize=(16, 8), dpi=80)
    # plt.title(mode)
    plt.subplot(4, 2, 1)
    plt.title('Change N:')
    plt.plot(x, '-', color='C0', label='Signal')
    plt.xlim([0, samples - 1])
    plt.legend(loc='upper right', fontsize=14)
    plt.grid(True)

    for j in range(len(r)):
        plt.subplot(4, 2, 2 + j)
        if j == 0:
            plt.title('Change R:')
        plt.stem(zz[j],
                 # use_line_collection=True,
                 linefmt='C2',
                 basefmt='C0',
                 label=f'R = {r[j]}, N = {n[j]}'
                 )
        plt.grid(True)
        plt.legend(loc='upper right', fontsize=14)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Number os samples
    N = 30

    # Filter parameters (length of lists should be same):
    flt_r = [2, 3, 3, 3, 5, 3, 7]
    flt_n = [3, 1, 3, 2, 3, 6, 3]

    plot_filter(r=flt_r, n=flt_n, samples=N, mode='Interpolator')
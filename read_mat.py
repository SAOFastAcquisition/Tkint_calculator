import numpy as np
from pymatreader import read_mat
import pandas as pd
from scipy import fftpack
import matplotlib.pyplot as plt

path = r'/home/anatoly/Yandex.Disk/Amp_ML/Hologramm/WorkHolDiff/HolDatN.mat'

if __name__ == '__main__':

    data = read_mat(path)
    keys_dict = data.keys()
    my_df = pd.DataFrame(data['data'])
    pass
    holo = my_df.iloc[138]['Holo1']

    A = fftpack.fft(holo)
    phy = np.arcsin(np.imag(A) / np.absolute(A))

    plt.plot(np.absolute(A))
    plt.show()
    plt.plot(phy)
    plt.show()
    pass
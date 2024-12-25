import numpy as np
from itertools import islice
import matplotlib.pyplot as plt
from scipy import fftpack
from cic_filter import CicFilter


def get_complex_holo(_txt_in):
    _list_txt = _txt_in.split(',')
    _list_float = [float(s) for s in _list_txt]
    _list_real = list(islice(_list_float, 0, len(_list_float), 2))
    _list_imag = list(islice(_list_float, 1, len(_list_float), 2))
    _holo_complex = np.array([complex(_list_real[i], _list_imag[i]) for i in range(len(_list_imag))])
    return _holo_complex


def get_mean_holo(_data, _mean_deg):
    _len = len(_data)
    _n = int(_len / _mean_deg)
    _data_mean = [np.mean(_data[i * _mean_deg:(i+1) * _mean_deg]) for i in range(_n)]
    return _data_mean


def get_holo_scale(_data):
    """

    :param _data:
    :return: _x - значение sin угла направления на точку отсчета ДПФ относительно фокуса параболы
            _holo - голограмма
    """
    _holo = _data['Holo1']
    _len_holo = len(_holo)
    _lambda = _data['Lambda']  # длина волны
    # _lambda = 3.95e-2

    _holo_width = _len_holo * _v_car / 64  # длина голограммы
    _r, _psy_ref = calc_antenna(_phy_ref + pi, 0)
    _psy_ref -= pi  # _psy_ref - угол на опорный щит из фокуса относительно оси АС
    _x = np.array([i for i in range(_len_holo)])  # номера отсчетов

    _x = _x * _lambda / _holo_width / 2

    return _x


if __name__ == '__main__':
    with open("2024-09-25-holo-cont-1.log", encoding='utf-8') as r_file:
        holo_txt = r_file.read()

    holo_complex = get_complex_holo(holo_txt)
    holo_mean = get_mean_holo(holo_complex, 16)
    holo_ampl = [np.abs(s) for s in holo_mean]
    plt.plot(holo_ampl)
    plt.show()
    # holo_phase = [np.angle(s) for s in holo_mean]
    A = np.fft.fftshift(fftpack.fft(holo_mean))
    # phy = np.arcsin(np.imag(A) / np.absolute(A))
    A_abs = np.absolute(A)
    r = 4
    n = 2
    clf_a = CicFilter(A_abs)
    clf_ai = clf_a.interpolator(r, n, mode=True) / r / r

    phy = np.unwrap(np.angle(A))
    phy_uw = CicFilter(phy)
    clf_ph = phy_uw.interpolator(r, n, mode=True) / r / r

    plt.plot(clf_ai)
    plt.show()

    plt.plot(clf_ph)
    plt.show()

print(holo_complex)
pass


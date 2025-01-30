import numpy as np
import scipy
from itertools import islice
import matplotlib.pyplot as plt
from scipy import fftpack
from cic_filter import CicFilter


p = 263.6  # Параметр параболы антенной системы (АС)
centers_distance = 156.2  # расстояние между геометрическим центром телескопа и фокусом антенны
angle_per_panel = 24 / 60 * np.pi / 180  # Размер щита в радианах 0.006981317007977318
n_center = 150  # Центральный щит АС
n_left = 100  # Левый край отображаемого сектора
n_right = 200  # Правый край отображаемого сектора
pi = np.pi
holo_width = 1.473  # meter


def get_initial_data(_path):
    if int(_path[-5]) < 3:
        _lambda1 = 2.17e-2
    else:
        _lambda1 = 1.73e-2

    with open(_path, encoding='utf-8') as r_file:
        _holo_txt = r_file.read()
    return _holo_txt, _lambda1

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


def get_holo_scale(_holo_width, _holo, _lambda):
    """
    :param  _holo_width: длина голограммы в метрах
            _len_holo: длина голограммы в отсчетах
            _lambda: длина волны
    :return: _x - значение sin угла направления на точку отсчета ДПФ относительно фокуса параболы
    """
    _len_holo = len(_holo)
    _x_scale = _lambda / _holo_width / 2  # 4 из-за двух отражений от зеркала основной антенны
    _x = np.array([(i - _len_holo // 2) * _x_scale for i in range(_len_holo)])  # номера отсчетов

    return _x


def calc_antenna(a, _angle0_d):
    """Возвращает радиальное положение щита относительно геометрического центра телескопа и направление на щит
    из фокуса АС при заданном направлении на щит из геометрического центра телескопа (ГЦТ)

    :param a: направление на щит/щиты из ГЦТ
    :param _angle0_d: смещение ДН АС по азимуту
    :return:
    """

    def func_under_solve(_angle1):
        """
        Целевая функция для нахождения углового положения центра щита относительно фокуса параболы _angle1 при его
        заданной угловой координате относильно геометрического центра радиотелескопа a
        :param _angle1: угол относительно фокуса параболы как переменная
        :return: угол относильно геометрического центра параболы радиотелескопа за вычетом заданного угла a
        относительно геометрического центра - целевая функция для нахождения angle1, соответствующего а
        """
        dr = centers_distance
        _r1 = p / (1 - np.cos(_angle1))
        _r2 = np.sqrt(_r1 ** 2 + dr ** 2 - 2 * _r1 * dr * np.cos(_angle1))
        _coeff = _r1 / _r2
        _f = np.pi - np.arcsin(np.sin(_angle1) * _coeff) - a
        return _f

    try:
        _leng = len(a)
    except:
        _leng = 1

    z = scipy.optimize.root(func_under_solve, x0=[pi] * _leng)
    parabola_arg = z.x
    parabola_value = p / (1 - np.cos(parabola_arg))

    _radius_out = np.sqrt(parabola_value ** 2 + centers_distance ** 2 -
                          2 * parabola_value * centers_distance * np.cos(parabola_arg))

    return _radius_out, parabola_arg + _angle0_d / 180 * pi


def get_holo_centered(_holo):

    _len_holo = len(_holo)
    _holo_ampl = np.array([np.abs(s) for s in _holo])
    _clf_a = CicFilter(_holo_ampl)
    _r = 1
    _n = 6
    _clf_ai = _clf_a.interpolator(_r, _n, mode=True) / _r / _r
    # plt.plot(_clf_ai)
    # plt.show()
    _max_index = _clf_ai.argmax()
    _index_c = _len_holo // 2
    _dn = _index_c - _max_index
    _holo_c = np.zeros(_len_holo + 1, dtype=complex)
    if _dn >= 0:
        _holo_c[_dn:] = _holo[0: _len_holo - _dn + 1]
    if _dn < 0:
        _holo_c[0:_len_holo - _dn + 1:] = _holo[-_dn-1: _len_holo]
    _holo_ampl = [np.abs(s) for s in _holo_c]

    return _holo_c


def get_panel_edges(_psy_centers):
    """
    Функция возвращает синусы углов из фокуса параболы на края щитов
    :param _psy_centers: углы на центры щитов из геометрического центра АС в рад
    :return _ksy_edges: синусы углов на края щитов из фокуса параболы
    """
    rad, _angles = calc_antenna(_psy_centers + pi, 0)
    _angles -= pi
    _angles = np.sin(_angles)  # Углы на центры щитов из фокуса АС

    # Вычисление синусов углов на края щитов из фокуса параболы
    _ksy_edges = [0] * (len(_angles) + 1)
    _ksy_edges[1:-1] = [(_angles[i + 1] + _angles[i]) / 2 for i in range(len(_angles) - 1)]
    _ksy_edges[0] = (_angles[0] - (_angles[1] - _angles[0]) / 2)
    _ksy_edges[-1] = (_angles[-1] + (_angles[-1] - _angles[-2]) / 2)
    return _ksy_edges, _angles


def simple_plot(_x, _y):
    plt.plot(_x, _y)
    plt.show()


if __name__ == '__main__':

    file_name = "2024-09-25-holo-cont-3.log"
    mean_deg = 4
    r = 4  # Коэффициент интерполяции
    n = 1  # Порядок фильтра

    holo_txt, lambda1 = get_initial_data(file_name)
    holo_complex = get_complex_holo(holo_txt)
    holo_mean = get_mean_holo(holo_complex, mean_deg)
    holo_mean_c = get_holo_centered(holo_mean)

    ksy_scale = get_holo_scale(holo_width, holo_mean_c, lambda1)         # Шкала в пространстве синусов углов параболы
    x_scale = np.linspace(-holo_width / 2, holo_width / 2, len(holo_mean_c))     # Шкала в пространстве волновых векторов
    psy_centers = np.linspace(-50 * angle_per_panel, 50 * angle_per_panel, 101)  # Углы на Центры щитов из
                                                                                    # геометрического центра АС
    ksy_edges, ksy_centers = get_panel_edges(psy_centers)       # Вычисление синусов углов на края и на центры щитов
                                                                # из фокуса параболы

    holo_ampl = np.absolute(holo_mean_c)

    A = np.fft.fftshift(fftpack.fft(holo_mean_c))
    A_abs = np.absolute(A)
    phy = np.unwrap(np.angle(A))
    simple_plot(ksy_scale, phy)

    clf_a = CicFilter(A_abs)
    phy_uw = CicFilter(phy)
    clf_ai = clf_a.interpolator(r, n, mode=True) / r / r
    clf_ph = phy_uw.interpolator(r, n, mode=True) / r / r * lambda1 / 360 / 4  # meter
    ksy_scale_i = np.linspace(np.min(ksy_scale), np.max(ksy_scale), len(holo_mean_c) * r)
    ampl_av = [0] * len(psy_centers)
    phase_av = [0] * len(psy_centers)
    for i in range(len(psy_centers)):
        index = np.where((ksy_edges[i] <= ksy_scale_i) & (ksy_scale_i < ksy_edges[i + 1]))
        ampl_av[i] = np.mean(clf_ai[index])
        phase_av[i] = np.mean(clf_ph[index])
    num = [s / angle_per_panel + 150 for s in psy_centers]
    simple_plot(psy_centers / angle_per_panel + 150, ampl_av)
    simple_plot(psy_centers / angle_per_panel + 150, phase_av)
    # simple_plot(x_scale, holo_ampl)
    # simple_plot(ksy_scale_i, clf_ai)
    # simple_plot(ksy_scale_i, clf_ph)

    pass



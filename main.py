# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

p = 263.6                   # Параметр параболы антенной системы (АС)
centers_distance = 156.2    # расстояние между геометрическим центром телескопа и фокусом антенны
angle_per_panel = 24 / 60 * np.pi / 180     # Размер щита в радианах
panel_speed = 1.44e-3       # Скорость опорного щита в направлении геометрического центра телескопа (ГЦТ)
n_center = 100              # Центральный щит АС
n_left = 100                # Левый край отображаемого сектора
n_right = 233               # Правый край отображаемого сектора
pi = np.pi


def convert_polar_to_polar(_radius, _angle):
    """
    Функция возвращает положение щита в полярной системе, связанной с геометрическим центром телескопа по известному
    положению щита (_radius и _angle)в полярной системе, связанной с фокусом антенны (_radius_out, _angle_out)
    :param _radius:
    :param _angle:
    :return: положение щита в полярной системе, связанной с фокусом антенны
    """
    _dR = centers_distance
    _radius_out = np.sqrt(_radius ** 2 + _dR ** 2 - 2 * _radius * _dR * np.cos(_angle))
    _coeff = _radius / _radius_out
    _angle_out = np.pi - np.arcsin(np.sin(_angle) * _coeff)
    return _radius_out, _angle_out


def panel_angles(_n_left, _n_right):
    _n_sector = _n_right - _n_left + 1
    _sector_left_pos = (150 - _n_left + 1 / 2) * angle_per_panel + pi
    _sector_right_pos = (_n_right - 150 + 1 / 2) * angle_per_panel + pi
    _panel_angle_pos = np.linspace((_sector_left_pos - angle_per_panel / 2),
                                   -(_sector_right_pos - angle_per_panel / 2),
                                   _n_sector) + np.pi
    return _panel_angle_pos


def plot_polar(_radius, _angle):
    r = _radius
    theta = _angle
    plt.polar(theta, r)

    plt.show()


def plotly_polar(_theta, _radial, _plotly_dict):
    _theta = _theta / 2 / np.pi * 360
    _fig = px.scatter_polar(
        r=_radial,
        theta=_theta,
        start_angle=0,
        direction="counterclockwise",
        range_theta=_plotly_dict['range_theta'],
        range_r=[0, 300]
    )

    _fig.update_layout(
        polar=_plotly_dict['polar']
    )

    # fig.write_html('pic_sign', auto_open=True)
    _fig.show(auto_open=True)
    return _fig


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


def ref_panel_speed(_n_ref, _n_center):
    """
    Возвращает скорость движения опорного щита по направлению к фокусу АС
    :param _n_ref: номер опорного щита
    :param _n_center: номер центрального щита параболы АС
    :return: скорость движения опорного щита по направлению к фокусу АС в м/сек
    """
    _theta_n_ref = pi + (_n_center - _n_ref) * angle_per_panel
    _theta_n_center = pi + (150 - _n_center) * angle_per_panel
    _r, _phy_n_ref = calc_antenna(_theta_n_ref, _theta_n_center)

    _delta_angle = _phy_n_ref - _theta_n_ref
    _panel_speed_parabola = panel_speed * np.cos(_delta_angle)
    return _panel_speed_parabola


if __name__ == '__main__':
    angle0_d = -20  # Азимут в градусах
    n_center_az = int(150 + angle0_d / 24 * 60)
    ref_speed = ref_panel_speed(118, 150)
    ang = panel_angles(n_left, n_right)
    radius_out, arg = calc_antenna(ang, angle0_d)
    angle0 = angle0_d / 180 * pi
    plotly_dict = {'range_theta': [140 - angle0_d, 220 - angle0_d],
                   'polar': {
                       "angularaxis": {
                           "tickmode": "array",
                           "tickvals": list(range(0, 360, 360 // 90)),
                           "ticktext": [f"{a - 300}" for a in range(900, 0, -900 // 90)],
                       }
                   }
                   }
    fig = plotly_polar(ang - angle0, radius_out, plotly_dict)
    pass

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

p = 263.6
centers_distance = 156.2  # расстояние между геометрическим центром телескопа и фокусом антенны
angle_per_panel = 24 / 60 * np.pi / 180
panel_speed = 1.44e-3
n_left = 100
n_right = 233
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


def panel_angles():
    _n_sector = n_right - n_left + 1
    _sector_left_pos = (150 - n_left + 1 / 2) * angle_per_panel
    _sector_right_pos = (n_right - 150 + 1 / 2) * angle_per_panel
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
        range_theta=_plotly_dict['range_theta']
    )

    _fig.update_layout(
        polar=_plotly_dict['polar']
    )

    # fig.write_html('pic_sign', auto_open=True)
    _fig.show(auto_open=True)
    return _fig


def calc_antenna(a, _angle0_d):
    try:
        leng = len(a)
    except:
        leng = 1

    def func_under_solve(_angle1):
        """
        Целевая функция для нахождения углового положения центра щита относительно фокуса параболы _angle1 при его заданной
        угловой координате относильно геометрического центра радиотелескопа _angle2
        :param _angle1: угол относительно фокуса параболы
        :return: угол относильно геометрического центра радиотелескопа
        """
        dr = centers_distance
        _r1 = p / (1 - np.cos(_angle1))
        _r2 = np.sqrt(_r1 ** 2 + dr ** 2 - 2 * _r1 * dr * np.cos(_angle1))
        _coeff = _r1 / _r2
        _f = np.pi - np.arcsin(np.sin(_angle1) * _coeff) - a
        return _f

    f = func_under_solve
    z = scipy.optimize.root(f, x0=[pi] * leng)
    parabola_arg = z.x
    parabola_value = p / (1 - np.cos(parabola_arg))

    radius_out = np.sqrt(parabola_value ** 2 + centers_distance ** 2 -
                         2 * parabola_value * centers_distance * np.cos(parabola_arg))
    plotly_dict = {'range_theta': [140 - _angle0_d, 220 - _angle0_d],
                   'polar': {
                       "angularaxis": {
                           "tickmode": "array",
                           "tickvals": list(range(0, 360, 360 // 90)),
                           "ticktext": [f"{a - 300}" for a in range(900, 0, -900 // 90)],
                       }
                   }
                   }

    return radius_out, parabola_arg, plotly_dict


def ref_panel_speed(_n_ref, _n_center):
    _theta_n_ref = pi + (_n_center - _n_ref) * angle_per_panel
    _theta_n_center = pi + (150 - _n_center) * angle_per_panel
    _r, _phy_n_ref, dict = calc_antenna(_theta_n_ref, _theta_n_center)

    _delta_angle = _phy_n_ref - _theta_n_ref
    _panel_speed_parabola = panel_speed * np.cos(_delta_angle)
    return _panel_speed_parabola


if __name__ == '__main__':
    angle0d = -0  # Азимут в градусах
    ref_speed = ref_panel_speed(118, 150)
    ang = panel_angles()
    radius_out, arg, plotly_dict = calc_antenna(ang, angle0d)
    angle0 = angle0d / 180 * pi
    fig = plotly_polar(panel_angles() - angle0, radius_out, plotly_dict)
    pass

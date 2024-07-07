import dash
from dash import html
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

p = 263.6
centers_distance = 156.2  # расстояние между геометрическим центром телескопа и фокусом антенны
angle_per_panel = 24 / 60 * np.pi / 180
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


def func_under_solve(_angle1):
    """
    Целевая функция для нахождения углового положения центра щита относительно фокуса параболы _angle1 при его заданной
    угловой координате относильно геометрического центра радиотелескопа _angle2
    :param _angle1: угол относительно фокуса параболы
    :return: угол относильно геометрического центра радиотелескопа
    """
    _angle2 = panel_angles()
    dr = centers_distance
    _r1 = p / (1 - np.cos(_angle1))
    _r2 = np.sqrt(_r1 ** 2 + dr ** 2 - 2 * _r1 * dr * np.cos(_angle1))
    _coeff = _r1 / _r2
    _f = np.pi - np.arcsin(np.sin(_angle1) * _coeff) - _angle2
    return _f


def panel_angles():
    _n_sector = n_right - n_left + 1
    _sector_left_pos = (150 - n_left + 1 / 2) * angle_per_panel
    _sector_right_pos = (n_right - 150 + 1 / 2) * angle_per_panel
    _panel_angle_pos = np.linspace((_sector_left_pos - angle_per_panel / 2),
                                   -(_sector_right_pos - angle_per_panel / 2),
                                   _n_sector) + np.pi
    return _panel_angle_pos


def plotly_polar(_theta, _radial, _plotly_dict):
    _theta = _theta / 2 / np.pi * 360
    fig = px.scatter_polar(
        r=_radial,
        theta=_theta,
        start_angle=0,
        direction="counterclockwise",
        range_theta=_plotly_dict['range_theta']
    )

    fig.update_layout(
        polar=_plotly_dict['polar']
    )

    # fig.write_html('pic_sign', auto_open=True)
    fig.show()


def calc_antenna(_angle0_d):
    angle0 = _angle0_d / 180 * pi
    # angle = np.linspace(pi * (1 - 0.25), pi * (1 + 0.25), 100)
    # r1 = p / (1 - np.cos(angle))

    # r2, angle2 = convert_polar_to_polar(r1, angle)
    # plot_polar(r2, angle2 - pi / 8)
    z = scipy.optimize.root(func_under_solve, x0=[pi] * (n_right - n_left + 1))
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
    plotly_polar(panel_angles() - angle0, radius_out, plotly_dict)


#           *** APPLICATION ***
app = dash.Dash(name='Hologram Application')

app.layout = html.Div([html.H2('Holography')])
angle0 = 24
calc_antenna(angle0)

if __name__ == '__main__':
    app.run_server(debug=True)

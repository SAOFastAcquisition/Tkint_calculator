import dash
from dash import html
import numpy as np
import pandas as pd
from pathlib import Path
import scipy.optimize
from scipy import fftpack
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from pymatreader import read_mat

p = 263.6  # Параметр параболы антенной системы (АС)
centers_distance = 156.2  # расстояние между геометрическим центром телескопа и фокусом антенны
angle_per_panel = 24 / 60 * np.pi / 180  # Размер щита в радианах
panel_speed = 1.44e-3  # Скорость опорного щита в направлении геометрического центра телескопа (ГЦТ)
n_center = 100  # Центральный щит АС
n_left_spect = 100  # Левый край отображаемого сектора
n_right_spect = 233  # Правый край отображаемого сектора
pi = np.pi


def path_to_data():
    """
    Определяет путь на конкретной машине к корню каталога данных.
    """
    head_path1 = Path(r'D:\YandexDisk\Amp_ML\Hologramm\WorkHolDiff')  # Путь к каталогу данных для домашнего ноута
    head_path1a = Path(
        r'E:\YandexDisk\Amp_ML\Hologramm\WorkHolDiff\HolDatN.mat')  # Путь к каталогу данных для домашнего ноута
    head_path1b = Path(
        r'G:\YandexDisk\Amp_ML\Hologramm\WorkHolDiff\HolDatN.mat')  # Путь к каталогу данных для домашнего ноута
    head_path2 = Path(
        r'/media/anatoly/Samsung_T5/YandexDisk/Amp_ML/Hologramm/WorkHolDiff')  # Путь к каталогу данных для рабочего компа
    head_path2a = Path(
        r'/media/anatoly/T7/YandexDisk/Amp_ML/Hologramm/WorkHolDiff')  # Путь к каталогу данных для рабочего компа
    head_path3 = Path(r'D:\YandexDisk\Amp_ML\Hologramm\WorkHolDiff\HolDatN.mat')  # Путь к каталогу данных для ноута ВМ
    head_path4 = Path(
        r'J:\YandexDisk\Amp_ML\Hologramm\WorkHolDiff\HolDatN.mat')  # Путь к каталогу данных для notebook 'Khristina'

    if head_path1.is_dir():
        head_path_out = head_path1
    elif head_path1a.is_dir():
        head_path_out = head_path1a
    elif head_path1b.is_dir():
        head_path_out = head_path1b
    elif head_path2.is_dir():
        head_path_out = head_path2
    elif head_path2a.is_dir():
        head_path_out = head_path2a
    elif head_path3.is_dir():
        head_path_out = head_path3
    elif head_path4.is_dir():
        head_path_out = head_path4
    else:
        return 'Err'
    return Path(head_path_out, 'HolDatN.mat')


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
    _angle2 = get_panel_angles()
    dr = centers_distance
    _r1 = p / (1 - np.cos(_angle1))
    _r2 = np.sqrt(_r1 ** 2 + dr ** 2 - 2 * _r1 * dr * np.cos(_angle1))
    _coeff = _r1 / _r2
    _f = np.pi - np.arcsin(np.sin(_angle1) * _coeff) - _angle2
    return _f


def get_panel_angles(_n_left, _n_right):
    _n_sector = _n_right - _n_left + 1
    _sector_left_pos = (150 - _n_left + 1 / 2) * angle_per_panel + pi
    _sector_right_pos = -(_n_right - 150 + 1 / 2) * angle_per_panel + pi
    _panel_angle_pos = np.linspace((_sector_left_pos - angle_per_panel / 2),
                                   (_sector_right_pos - angle_per_panel / 2),
                                   _n_sector)
    return _panel_angle_pos


def get_matlab_data(_path, _loc):
    _data = read_mat(_path)

    # _keys_dict = data.keys()
    _data_df = pd.DataFrame(_data['data'])
    try:
        _data_row = _data_df.iloc[_loc]
    except IndexError:
        exit(f'Error in get_matlab_data(_path, _loc). Positional index {_loc} is wrong. Data not have. Change index')
    return _data_row


def get_holo_scale(_data):
    """

    :param _data:
    :return: _x - значение sin угла направления на точку отсчета ДПФ относительно фокуса параболы
            _holo - голограмма
    """
    _holo = _data['Holo1']
    _len_holo = len(_holo)
    _v_car = _data['V_car']  # скорость каретки
    _v_sh = _data['V_sh']  # скорость щита в направлении на ГЦА
    _lambda = _data['Lambda']  # длина волны
    # _lambda = 3.95e-2

    _holo_width = _len_holo * _v_car / 64  # длина голограммы
    _n_ref = _data['Nref']  # опорный щит
    _v_sh_f = ref_panel_speed(_n_ref, 150)  # скорость щита в направлении на фокус АС
    _phy_ref = (150 - _n_ref) * 24 / 60 * np.pi / 180  # Угол на опорный щит из ГЦА
    _r, _psy_ref = calc_antenna(_phy_ref + pi, 0)
    _psy_ref -= pi  # _psy_ref - угол на опорный щит из фокуса относительно оси АС
    _x = np.array([i for i in range(_len_holo)])  # номера отсчетов
    _с0 = (1 + np.cos(_phy_ref)) * _v_sh_f / _v_car  # сдвиг восстановленного поля из-за движения опорного щита
    if _n_ref > 150:
        # Масштабирование отсчетов голограммы к синусу угла
        # из фокуса АС относительно оси АС и учет сдвига из-за положения и движения опорного щита
        _x = _x * _lambda / _holo_width - _с0 + np.sin(_psy_ref)
    else:
        _x = _x * _lambda / _holo_width - _с0 - np.sin(_psy_ref)

    return _x


def get_field(_path, record_num):
    registration = get_matlab_data(_path, record_num)
    holo = registration['Holo1']
    n_left = registration['Nl']
    n_right = registration['Nr']
    x = get_holo_scale(registration)

    angle01 = get_panel_angles(n_left, n_right)
    r, angle_p = calc_antenna(angle01, 5)

    # print(f'angle_p = {angle_p}')
    angle_p -= pi
    angle_p = np.arcsin(angle_p)
    # print(f'angle_p1 = {angle_p}')
    # n = 150 + np.arcsin(x) * 180 / np.pi / 24 * 60

    A = fftpack.fft(holo)
    phy = np.arcsin(np.imag(A) / np.absolute(A))
    dict_ampl = {'y_title': 'Amplitude, arb. units',
                 'x_title': 'Panel Number',
                 'title': 'Amplitude',
                 'ticktext': [f"{a}" for a in range(n_right, n_left - 2, -2)],
                 'tickvals': angle_p[::2],
                 'xrange': [-0.4, 0.4],
                 'yrange': [0, 3e6]
                 }

    dict_phase = {'y_title': 'Phase, rad',
                  'x_title': 'Panel Number',
                  'title': 'Phase',
                  'ticktext': [f"{a}" for a in range(n_right, n_left - 2, -2)],
                  'tickvals': angle_p[::2],
                  'xrange': [-0.4, 0.4],
                  'yrange': [-2, 2]
                  }
    return x, A, phy, dict_ampl, dict_phase


def plotly_form(*args):
    _x, _y, _dict = args

    _fig = go.Figure()
    # _m = len(_data)
    # _l = 0
    # for i in range(_m):
    #     if len(_data[i][2]) > 2:
    _fig.add_trace(go.Scatter(x=_x, y=_y))
    # _l += 1

    _fig.update_layout(title=_dict['title'], title_x=0.5,
                       xaxis_title=_dict['x_title'],
                       yaxis_title=_dict['y_title'],
                       xaxis={
                           "tickmode": "array",
                           "tickvals": _dict['tickvals'],
                           "ticktext": _dict['ticktext']
                       },
                       # hovermode="x",
                       paper_bgcolor='rgba(165,225,225,0.5)',
                       plot_bgcolor='rgba(90,150,60,0.25)')
    _fig.update_xaxes(range=_dict['xrange'])
    _fig.update_yaxes(range=_dict['yrange'])
    # _fig.add_annotation(text=_info[0], xref="paper", yref="paper", x=1, y=1.17, showarrow=False)
    # _fig.add_annotation(text=_info[1], xref="paper", yref="paper", x=1, y=1.12, showarrow=False)
    # _fig.add_annotation(text=_info[4], xref="paper", yref="paper", x=1, y=1.07, showarrow=False)

    # _fig.update_yaxes(type=_dict['scale'],
    #                   ticks="inside",
    #                   showline=True, linewidth=1.25, linecolor='black',
    #                   showgrid=True, gridwidth=0.75, gridcolor='LightPink')
    # _fig.update_xaxes(ticks="inside",
    #                   showline=True, linewidth=1, linecolor='black',
    #                   showgrid=True, gridwidth=1, gridcolor='LightPink')

    # _fig.write_html(_dict['pic_sign'], auto_open=True)
    _fig.show(auto_open=True)
    # return _fig


def plotly_polar(_theta, _radial, _plotly_dict):
    _theta = _theta / 2 / np.pi * 360

    _fig = px.scatter_polar(
        r=_radial,
        theta=_theta,
        start_angle=0,
        direction="counterclockwise",
        range_theta=_plotly_dict['range_theta'],
        range_r=_plotly_dict['range_r']

    )

    _fig.update_layout(
        polar=_plotly_dict['polar']
    )

    # fig.write_html('pic_sign', auto_open=True)
    _fig.show(auto_open=True)
    return _fig


def calc_antenna(a, angle0_d):
    """Возвращает радиальное положение щита относительно геометрического центра телескопа и направление на щит
    из фокуса АС при заданном направлении на щит из геометрического центра телескопа (ГЦТ)

    :param a: направление на щит/щиты из ГЦТ
    :param _angle0_d: смещение ДН АС по азимуту
    :return:
    """
    angle0_d = angle0_d / 60 * pi / 180

    def func_under_solve(_angle1):
        """
        Целевая функция для нахождения углового положения центра щита относительно фокуса параболы _angle1 при его
        заданной угловой координате относильно геометрического центра радиотелескопа a
        :param _angle1: угол относительно фокуса параболы как переменная
        :return: угол относильно геометрического центра параболы радиотелескопа за вычетом заданного угла a
        относительно геометрического центра - целевая функция для нахождения angle1, соответствующего а
        """
        dr = centers_distance
        _r1 = p / (1 - np.cos(_angle1 + angle0_d))
        _r2 = np.sqrt(_r1 ** 2 + dr ** 2 - 2 * _r1 * dr * np.cos(_angle1))
        _coeff = _r1 / _r2
        _f = np.pi - np.arcsin(np.sin(_angle1 + angle0_d) * _coeff) - a
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

    return _radius_out, parabola_arg


def get_antenna(_n_left_spect, _n_right_spect, angle0):
    n_center_az = int(150 + angle0 / 24 * 60)
    ang = get_panel_angles(_n_left_spect, _n_right_spect)
    radius_out, arg = calc_antenna(ang, angle0)
    angle0 = angle0 / 180 * pi
    plotly_dict = {'range_theta': [140 - angle0, 220 - angle0],
                   'range_r': [0, 300],
                   'polar': {
                       "angularaxis": {
                           "tickmode": "array",
                           "tickvals": list(range(0, 360, 360 // 90)),
                           "ticktext": [f"{a - 300}" for a in range(900, 0, -900 // 90)],
                       }
                   }
                   }
    return ang - angle0, radius_out, plotly_dict


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


#           *** APPLICATION ***
app = dash.Dash(name='Hologram Application')

angle0 = 0
theta, r, plotly_dict = get_antenna(n_left_spect, n_right_spect, angle0)
path = path_to_data()
x, A, phy, dict_ampl, dict_phase = get_field(path, 127)

# fig = plotly_polar(theta, r, plotly_dict)
plotly_form(np.arcsin(x), np.absolute(A), dict_ampl)
plotly_form(np.arcsin(x), phy, dict_phase)

# app.layout = html.Div([html.H2('Holography')])

if __name__ == '__main__':
    app.run_server(debug=True)

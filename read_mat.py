import numpy as np
from pymatreader import read_mat
import pandas as pd
from scipy import fftpack
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go
from holo_calc import calc_antenna, ref_panel_speed

path = r'D:\YandexDisk\Amp_ML\Hologramm\WorkHolDiff\HolDatN.mat'
pi = np.pi
angle_per_panel = 24 / 60 * np.pi / 180
n_left = 130
n_right = 170


def get_panel_angles():
    _n_sector = n_right - n_left + 1
    _sector_left_pos = (150 - n_left + 1 / 2) * angle_per_panel + pi
    _sector_right_pos = (- n_right + 150 + 1 / 2) * angle_per_panel + pi
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
    _holo = registration['Holo1']
    _len_holo = len(_holo)
    _v_car = registration['V_car']                      # скорость каретки
    _v_sh = registration['V_sh']                        # скорость щита в направлении на ГЦА
    _lambda = registration['Lambda']  # длина волны
    # _lambda = 3.95e-2

    _holo_width = _len_holo * _v_car / 64               # длина голограммы
    _n_ref = registration['Nref']                       # опорный щит
    _v_sh_f = - ref_panel_speed(_n_ref, 150)            # скорость щита в направлении на фокус АС
    _phy_ref = (150 - _n_ref) * 24 / 60 * np.pi / 180   # Угол на опорный щит из ГЦА
    _r, _psy_ref = calc_antenna(_phy_ref + pi, 0)
    _psy_ref -= pi                                      # _psy_ref - угол на опорный щит из фокуса относительно оси АС
    _x = np.array([i for i in range(_len_holo)])        # номера отсчетов
    _с0 = (1 + np.cos(_phy_ref)) * _v_sh_f / _v_car     # сдвиг восстановленного поля из-за движения опорного щита
    _x = _x * _lambda / _holo_width + _с0 - np.sin(_psy_ref)    # Масштабирование отсчетов голограммы к синусу угла
    # из фокуса АС относительно оси АС и учет сдвига из-за положения и движения опорного щита

    return _x


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


if __name__ == '__main__':

    registration = get_matlab_data(path, 138)
    holo = registration['Holo1']
    x = get_holo_scale(registration)
    angle0 = get_panel_angles()
    r, angle_p = calc_antenna(angle0, 1)
    angle_p -= pi
    # n = 150 + np.arcsin(x) * 180 / np.pi / 24 * 60

    A = fftpack.fft(holo)
    phy = np.arcsin(np.imag(A) / np.absolute(A))
    dict_ampl = {'y_title': 'Amplitude, arb. units',
                 'x_title': 'Panel Number',
                 'title': 'Amplitude',
                 'ticktext': [f"{a}" for a in range(170, 128, -2)],
                 'tickvals': angle_p[::2],
                 'xrange': [-0.4, 0.4],
                 'yrange': [0, 1e6]
                 }
    plotly_form(x, np.absolute(A), dict_ampl)
    dict_phase = {'y_title': 'Phase, rad',
                  'x_title': 'Panel Number',
                  'title': 'Phase',
                  'ticktext': [f"{a}" for a in range(170, 128, -2)],
                  'tickvals': angle_p[::2],
                  'xrange': [-0.4, 0.4],
                  'yrange': [-2, 2]
                  }
    plotly_form(np.arcsin(x), phy, dict_phase)
    pass

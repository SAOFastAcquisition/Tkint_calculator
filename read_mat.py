import numpy as np
from pymatreader import read_mat
import pandas as pd
from scipy import fftpack
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objs as go

path = r'D:\YandexDisk\Amp_ML\Hologramm\WorkHolDiff\HolDatN.mat'


def get_matlab_data(_path, _loc):
    _data = read_mat(_path)

    # _keys_dict = data.keys()
    _data_df = pd.DataFrame(_data['data'])
    try:
        _data_row = _data_df.iloc[_loc]
    except IndexError:
        exit(f'Error in get_matlab_data(_path, _loc). Positional index {_loc} is wrong. Data not have. Change index')
    return _data_row


def get_holo_param(_data):
    _holo = registration['Holo1']
    _len_holo = len(_holo)
    _v_car = registration['V_car']   # скорость каретки
    _v_sh = registration['V_sh']     # скорость щита
    _lambda = registration['Lambda']  # длина волны
    # _lambda = 3.3e-2
    _holo_width = _len_holo * _v_car / 64    # длина голограммы
    _n_ref = registration['Nref']    # опорный щит
    _phy_ref = (150 - _n_ref) * 24 / 60 * np.pi / 180   # Угол на опорный щит относительно оси антенны
    _x = np.array([i for i in range(_len_holo)])    # номера отсчетов
    _с0 = (1 + np.cos(_phy_ref)) * _v_sh / _v_car   # сдвиг восстановленного поля из-за движения опорного щита
    _x = _x * _lambda / _holo_width / 2 + _с0 / 2 - np.sin(_phy_ref)

    return _x, _holo

def plotly_form(*args):
    _x, _y = args

    _fig = go.Figure()
    # _m = len(_data)
    # _l = 0
    # for i in range(_m):
    #     if len(_data[i][2]) > 2:
    _fig.add_trace(go.Scatter(x=_x, y=_y))
            # _l += 1

    _fig.update_layout(title='Amplitude', title_x=0.5,
                       xaxis_title='angle',
                       yaxis_title='Amplitude, arb. units',
                       # hovermode="x",
                       paper_bgcolor='rgba(165,225,225,0.5)',
                       plot_bgcolor='rgba(90,150,60,0.25)')
    _fig.update_xaxes(range=[-0.2, 0.2])
    _fig.update_yaxes(range=[0, 1e6])
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
    x, holo = get_holo_param(registration)
    n = 150 + np.arcsin(x) * 180 / np.pi / 24 * 60

    A = fftpack.fft(holo)
    phy = np.arcsin(np.imag(A) / np.absolute(A))
    plotly_form(x, np.absolute(A))
    # plt.plot(np.absolute(A))
    # plt.show()
    plt.plot(x, phy)
    plt.show()
    pass
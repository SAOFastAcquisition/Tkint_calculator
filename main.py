# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt


def convert_polar_to_polar(_radius, _angle, _centers_distance):
    _dR = _centers_distance
    _radius_out = np.sqrt(_radius ** 2 + _dR ** 2 - 2 * _radius * _dR * np.cos(_angle))
    _coeff = _radius / _radius_out
    _angle_out = np.pi - np.arcsin(np.sin(_angle) * _coeff)
    return _radius_out, _angle_out


def func_under_solve(_angle1):
    _angle2 = np.linspace(-0.051, 0.251, 137)
    _angle2 = (_angle2 + 1) * np.pi
    _p = 263.6
    _dr = 156.2
    _r1 = _p / (1 - np.cos(_angle1))
    _r2 = np.sqrt(_r1 ** 2 + _dr ** 2 - 2 * _r1 * _dr * np.cos(_angle1))
    _coeff = _r1 / _r2
    _f = np.pi - np.arcsin(np.sin(_angle1) * _coeff) - _angle2
    return _f


def plot_polar(_radius, _angle):
    r = _radius
    theta = _angle
    plt.polar(theta, r)

    plt.show()


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    centers_distance = 156.2
    pi = np.pi
    angle0 = pi / 8
    angle = np.linspace(pi * (1 - 0.25), pi * (1 + 0.25), 100)
    r1 = 263.6 / (1 - np.cos(angle))
    # angle = angle - pi / 8
    # plot_polar(r1, angle)
    r2, angle2 = convert_polar_to_polar(r1, angle, centers_distance)
    # plot_polar(r2, angle2 - pi / 8)

    z = scipy.optimize.root(func_under_solve, x0=[pi] * 137)
    print(z.x)
    plt.plot( (np.linspace(-0.051, 0.251, 137) + 0), np.array(z.x) / pi - 1)
    plt.show()
    pass

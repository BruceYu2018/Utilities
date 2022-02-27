#! coding:utf-8
"""
colorMapJet.py
Created by fifi  (2016/04/02 22:43)
"""
__version__ = '0.0'


def getColorJet(v, vmin, vmax):
    c = [1., 1., 1.]  # white
    R, G, B = 0, 1, 2
    dv = 0

    if v < vmin:
        v = vmin
    if v > vmax:
        v = vmax
    dv = vmax - vmin

    if v < (vmin + 0.25 * dv):
        c[R] = 0
        c[G] = 4 * (v - vmin) / dv
    elif v < (vmin + 0.5 * dv):
        c[R] = 0.
        c[B] = 1 + 4 * (vmin + 0.25 * dv - v) / dv
    elif v < (vmin + 0.75 * dv):
        c[R] = 4 * (v - vmin - 0.5 * dv) / dv
        c[B] = 0.
    else:
        c[G] = 1 + 4 * (vmin + 0.75 * dv - v) / dv
        c[B] = 0.

    return c


def getColorJetRGBi(v, vmin, vmax):
    cf = getColorJet(v, vmin, vmax)
    return [int(c * 255) for c in cf]


def getColorJetRGBf(v, vmin, vmax):
    return getColorJet(v, vmin, vmax)


if __name__ == "__main__":
    for i in range(0, 256):
        print i / 255., getColorJetRGBi(v=i / 255., vmin=0, vmax=1)

    #for i in range(0, 10):
    #    print  i / 100., getColorJet(v=i / 100., vmin=0, vmax=1)

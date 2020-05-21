#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import absolute_import
from matplotlib import pyplot as plt
from matplotlib.offsetbox import AnchoredText
import numpy

location_numbers = {
    'upper right'  :    1,
    'upper left'   :    2,
    'lower left'   :    3,
    'lower right'  :    4,
    'right'        :    5,
    'center left'  :    6,
    'center right' :    7,
    'lower center' :    8,
    'upper center' :    9,
    'center'       :    10
}

def plot_lines(text, *args, **kwargs):
    patchopts = kwargs.pop('patchopts', None)
    sep     = kwargs.pop('separator', u'\n')
    if not isinstance(text, str):
        linefmt = kwargs.pop('linefmt', u'{}')
        if isinstance(linefmt, str) or isinstance(linefmt, unicode):
            fmt = linefmt
            linefmt = lambda *a: fmt.format(*a)
        lines = []
        lst = [linefmt(*line) if isinstance(line, (list, tuple)) else linefmt(line) for line in text]
        text = sep.join(lst)

    header, footer = kwargs.pop('header', None), kwargs.pop('footer', None)
    if header: text = header+sep+text
    if footer: text = text+sep+footer

    if kwargs.pop('dump', False):
        print('Text:\n', text, sep='')

    # if bbox is None: bbox = dict(facecolor='white', alpha=1)
    outside = kwargs.pop('outside', None)
    loc = kwargs.pop('loc', None)
    if outside:
        loc='upper left'
        kwargs.setdefault('borderpad', 0.0)
    if isinstance(loc, str):
        loc = location_numbers[loc]
    if not loc:
        loc = 1
    prop = kwargs.pop('prop', {})
    ax = plt.gca()
    if outside:
        kwargs[ 'bbox_to_anchor' ]=outside
        kwargs[ 'bbox_transform' ]=ax.transAxes

    at = AnchoredText(text, loc, *args, prop=prop, **kwargs)
    if patchopts:
        at.patch.set(**patchopts)

    ax.add_artist(at)
    return at

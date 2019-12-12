#!/usr/bin/env python

from __future__ import print_function
from sys import argv
from os import environ, path, makedirs

# Setup headless mode
if '--batch' in argv or not environ.get('DISPLAY', ''):
    import matplotlib
    matplotlib.use('Agg')

from mpl_tools.helpers import savefig
from gna.graphviz import savegraph

# Robust output filename generating routine
def tutorial_image_name(ext, suffix=''):
    outpath = 'output/tutorial_img'
    scriptpath = argv[0]
    marker = 'macro/tutorial/'

    start = scriptpath.find(marker)
    assert start>=0
    end = start+len(marker)
    scriptpath_sub = scriptpath[end:]

    scriptpath_path, scriptpath_base = path.split(scriptpath_sub)
    base, _ = path.splitext(scriptpath_base)

    outpath = path.join(outpath, scriptpath_path)

    if not path.exists(outpath):
        print('Create output folder:', outpath)
        makedirs(outpath)

    assert path.isdir(outpath), '{} is not a directory'.format(outpath)

    if suffix:
        base = '_'.join((base, suffix))

    return path.join(outpath, base+'.'+ext)

if __name__ == "__main__":
    argv[0] = '/home/gonchar/work/gna/gna/macro/tutorial/plotting/02_points_plot_vs.py'
    expect  = 'output/tutorial_img/plotting/02_points_plot_vs_test.png'
    t=tutorial_image_name('png', 'test')

    print(t)

    assert expect==t

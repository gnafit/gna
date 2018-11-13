from sys import argv
from os import environ
if '--batch' in argv or not environ.get('DISPLAY', ''):
    import matplotlib
    matplotlib.use('Agg')

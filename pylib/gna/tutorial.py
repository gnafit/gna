from sys import argv
if '--batch' in argv:
    import matplotlib
    matplotlib.use('Agg')

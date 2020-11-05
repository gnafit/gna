#!/usr/bin/env python3

import pickle
import numpy as np
from matplotlib import pyplot as plt
from mpl_tools.helpers import savefig

def main(opts):
    data = dict()
    for datum in opts.files:
        source = datum['info']['source']
        step   = datum['info']['step']
        fun    = datum['fun']

        data.setdefault(source, []).append( (step, fun) )

    fig = plt.figure()
    ax = plt.subplot(111, xlabel='Internal bin width, keV', ylabel=r'$\Delta \chi^2$', title='Fit stability')
    ax.minorticks_on()
    ax.grid()
    ax.set_xlim(0.0, 21.0)

    markers = {
            'nolsnl': 'x',
            'subst': '+',
            'proper': 'o',
            'dumb': 'o',
            }
    offset = min(y for (x,y) in data['subst'])
    for source, xy in data.items():
        x, y = np.array(xy).T
        idx = np.argsort(x)
        x, y = x[idx], y[idx]
        x*=1000.0
        # y-=offset

        ax.plot(x, y, markers.get(source, '*'), label=source, markerfacecolor='none')

        ax.legend(title='Mode:')
        savefig(opts.output, suffix='_'+source)

    plt.show()

def load(fname):
    with open(fname, 'rb') as f:
        ret=pickle.load(f, encoding='latin1')['fitresult']['min']
        assert ret['success']
        return ret

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('files', nargs='+', type=load, help='input files')
    parser.add_argument('-o', '--output', nargs='+', default=[], help='output file to write')

    main( parser.parse_args() )

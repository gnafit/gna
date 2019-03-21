#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import ROOT as R
from mpl_tools import root2numpy, root2mpl
root2numpy.bind()
root2mpl.bind()
from matplotlib import pyplot as plt
import numpy as N

def main(args):
    hist = args.input.Get(args.name)

    buf = hist.get_buffer()
    print('Buffer', buf.shape, buf)

    hist.pcolorfast(colorbar=True)

    buf1 = N.ma.array(buf, mask=buf==0.0)
    plt.matshow(buf1)

    plt.show()

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('input', type=lambda x: R.TFile(x, 'read'))
    parser.add_argument('-n', '--name', default='corrmap')

    main(parser.parse_args())

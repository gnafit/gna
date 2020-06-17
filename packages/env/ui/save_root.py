# -*- coding: utf-8 -*-
"""Save given path within env to the ROOT file"""

from __future__ import print_function
from gna.ui import basecmd
from pprint import pprint
from tools.dictwrapper import DictWrapper
from collections import OrderedDict
from load import ROOT as R

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('paths', nargs='+', help='paths to save')
        parser.add_argument('-v', '--verbose', action='count', help='be more verbose')
        parser.add_argument('-o', '--output', required=True, help='Output fil ename')

    def init(self):
        storage = self.env.future
        verbose = self.opts.verbose

        if verbose>1:
            print('Create output file:', self.opts.output)
        output = R.TFile(self.opts.output, 'RECREATE')
        for path in self.opts.paths:
            try:
                st = storage[path]
            except KeyError:
                raise Exception('Unable to read data path: '+path)

            saveobjects(output, st, verbose)

        if verbose:
            print('Save output file:', self.opts.output)
            if verbose>1:
                output.ls()
        output.Close()

def saveobjects(odir, obj, verbose):
    if isinstance(obj, R.TObject):
        if verbose>1:
            print('Write', str(obj))
        odir.WriteTObject(obj, obj.GetName(), 'overwrite')
        return

    if isinstance(obj, DictWrapper):
        for k, v in obj.walkitems():
            path, name = k[:-1], k[-1]
            if path:
                path = '/'.join(path)
                subdir = odir.GetDirectory(path) or odir.mkdir(path)
            else:
                subdir = odir
            saveobjects(subdir, v, verbose)
        return

    print('Unable to save the object to ROOT file, skip:', type(obj))

"""Save the outputs as ROOT objects: TH1D, TH2D, TGraph"""

from __future__ import print_function
from gna.ui import basecmd, append_typed, qualified
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from mpl_tools.helpers import savefig
import numpy as np
from tools.yaml import yaml_load
from gna.bindings import common
from gna.env import env
from mpl_tools import root2numpy
from load import ROOT as R

def unpack(output, *args, **kwargs):
    dtype = output.datatype()
    ndim  = len(dtype.shape)
    if dtype.kind==2:
        if ndim==1:
            return unpack_hist1(output, dtype, *args, **kwargs)
        # elif ndim==2:
            # return unpack_hist2(output, dtype, *args, **kwargs)
        else:
            raise ValueError('Invalid histogram dimension '+str(ndim))
    elif dtype.kind==1:
        # return unpack_points(output, dtype, *args, **kwargs)
        raise ValueError('Saving TArray is not implemented')

    raise ValueError('Uninitialized output')

def set_axes(obj, kwargs):
    xlabel = kwargs.pop('xlabel', None)
    if xlabel:
        obj.GetXaxis().SetTitle(xlabel)
    ylabel = kwargs.pop('ylabel', None)
    if ylabel:
        obj.GetYaxis().SetTitle(ylabel)

def unpack_hist1(output, dtype, kwargs={}):
    dtype = dtype or output.datatype()
    data = output.data()

    edges = np.array(dtype.edgesNd[0], dtype='d')
    widths = edges[1:]-edges[:-1]
    rel_offsets = np.fabs(widths-widths[0])/widths.max()

    name  = kwargs.pop('name', '')
    title = kwargs.pop('label', kwargs.pop('title', ''))

    if (rel_offsets<1.e-9).all():
        # Constant width histogram
        hist = R.TH1D(name, title, edges.size-1, edges[0], edges[-1])
    else:
        hist = R.TH1D(name, title, edges.size-1, edges)

    buffer = root2numpy.get_buffer_hist1(hist)
    buffer[:] = data
    hist.SetEntries(data.sum())

    set_axes(hist, kwargs)

    if kwargs:
        raise Exception('Unparsed options in extra arguments for TH1D')

    return hist

def unpack_graph(outputx, outputy, kwargs):
    dtypex = outputx.datatype()
    dtypey = outputy.datatype()
    if dtypex.kind!=1 or dtypey.kind!=1:
        raise TypeError('Data kind is not Points')
    if list(dtypex.shape)!=list(dtypey.shape):
        raise ValueError('Data has inconsistent shapes')

    datax = outputx.data()
    datay = outputy.data()

    graph = R.TGraph(datax.size, datax, datay)
    set_axes(graph, kwargs)
    graph.SetNameTitle(kwargs.pop('name', ''), kwargs.pop('label', kwargs.pop('title', '')))

    if kwargs:
        raise Exception('Unparsed options in extra arguments for TGraph')

    return graph

class cmd(basecmd):
    def __init__(self, *args, **kwargs):
        basecmd.__init__(self, *args, **kwargs)

    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('-s', '--root-source', default=(), help='root namespace to copy from')
        parser.add_argument('-t', '--root-target', default=(), help='root namespace to copy to')
        parser.add_argument('-c', '--copy', nargs='+', action='append', default=[], help='Data to read and address to write', metavar=('from', 'to'))
        parser.add_argument('-g', '--copy-graph', nargs='+', action='append', default=[], help='Data to read (x,y) and address to write', metavar=('x', 'y'))
        parser.add_argument('-v', '--verbose', action='count', help='verbosity')

    def run(self):
        source = self.env.future.child(self.opts.root_source)
        target = self.env.future.child(self.opts.root_target)
        if self.opts.verbose:
            print('Converting to ROOT')
        for copydef in self.opts.copy:
            if len(copydef)<2:
                raise Exception('Invalid number of `copy` arguments: '+str(len(copydef)))
            (frmpath, to), extra = copydef[:2], copydef[2:]
            try:
                iterator = source.walkitems(startfromkey=frmpath)
            except KeyError:
                raise Exception('Invalid path: '+str(frmpath))
            for key, obs in iterator:
                kwargs = yaml_load(extra)
                kwargs.setdefault('name', to.rsplit('.', 1)[-1])
                try:
                    data = unpack(obs, kwargs)
                except ValueError:
                    print('Skipping unsupported object:', '.'.join(key))
                    continue
                targetkey = (to,)+key
                target[targetkey] = data

                if self.opts.verbose:
                    print('    {}->{}'.format('.'.join((frmpath,)+key), '.'.join(targetkey)))

        for copydef in self.opts.copy_graph:
            if len(copydef)<3:
                raise Exception('Invalid number of `copy-graph` arguments: '+str(len(copydef)))
            (frmx, frmy, to), extra = copydef[:3], copydef[3:]
            try:
                obsx = source[frmx]
                obsy = source[frmy]
            except KeyError:
                print('Unable to find key:', frm)

            kwargs = yaml_load(extra)
            kwargs.setdefault('name', to.rsplit('.', 1)[-1])
            data = unpack_graph(obsx, obsy, kwargs)
            target[to] = data

def list_get(lst, idx, default):
    return lst[idx] if idx<len(lst) else default

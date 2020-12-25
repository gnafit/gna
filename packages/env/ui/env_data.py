"""Recursively saves outputs as dictionaries with numbers and meta."""

from gna.ui import basecmd, append_typed, qualified
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from mpl_tools.helpers import savefig
import numpy as np
from tools.yaml import yaml_load
from gna.bindings import common
from gna.env import env

def unpack(output):
    dtype = output.datatype()
    if dtype.kind==2:
        return unpack_hist(output, dtype)
    elif dtype.kind==1:
        return unpack_points(output, dtype)

    raise ValueError('Uninitialized output')

def unpack_common(output, dtype, **kwargs):
    dtype = dtype or output.datatype()

    ret = dict(
        data  = output.data().copy(),
        shape = np.array(dtype.shape),
        ndim  = len(dtype.shape)
        )
    ret.update(kwargs)
    return ret

def unpack_hist(output, dtype):
    return unpack_common(output, dtype, type='hist', edges=[np.array(e) for e in dtype.edgesNd])

def unpack_points(output, dtype):
    return unpack_common(output, dtype, type='points')

def unpack_graph(outputx, outputy):
    dtypex = outputx.datatype()
    dtypey = outputy.datatype()
    if dtypex.kind!=1 or dtypey.kind !=1:
        raise TypeError('Data kind is not Points')
    if list(dtypex.shape)!=list(dtypey.shape):
        raise ValueError('Data has inconsistent shapes')

    x = unpack_common(outputx, dtypex)
    y = unpack_common(outputy, dtypey)

    return dict(
            x = x['data'],
            y = y['data'],
            type='graph',
            ndim = x['ndim'],
            shape = x['shape']
            )

class cmd(basecmd):
    def __init__(self, *args, **kwargs):
        basecmd.__init__(self, *args, **kwargs)

    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('-s', '--root-source', default=(), help='root namespace to copy from')
        parser.add_argument('-t', '--root-target', default=(), help='root namespace to copy to')
        parser.add_argument('-c', '--copy', nargs='+', action='append', default=[], help='Data to read and address to write', metavar=('from', 'to'))
        parser.add_argument('-g', '--copy-graph', nargs='+', action='append', default=[], help='Data to read (x,y) and address to write', metavar=('x', 'y'))
        parser.add_argument('-v', '--verbose', action='count', default=0, help='verbosity')

    def run(self):
        source = self.env.future.child(self.opts.root_source)
        target = self.env.future.child(self.opts.root_target)

        if self.opts.verbose:
            print('Converting to numpy')
        for copydef in self.opts.copy:
            if len(copydef)<2:
                raise Exception('Invalid number of `copy` arguments: '+str(len(copydef)))
            (frmpath, to), extra = copydef[:2], copydef[2:]
            upd = yaml_load(extra)
            try:
                iterator = source.walkitems(startfromkey=frmpath)
            except KeyError:
                raise Exception('Invalid path: '+str(frmpath))
            for key, obs in iterator:
                try:
                    data = unpack(obs)
                except ValueError:
                    print('Skipping unsupported object:', '.'.join(key))
                    continue
                targetkey = (to,)+key
                target[targetkey] = data

                if self.opts.verbose:
                    print('  {}->{}'.format('.'.join((frmpath,)+key), '.'.join(targetkey)))

                data.update(upd)

        for copydef in self.opts.copy_graph:
            if len(copydef)<3:
                raise Exception('Invalid number of `copy-graph` arguments: '+str(len(copydef)))
            (frmx, frmy, to), extra = copydef[:3], copydef[3:]
            try:
                obsx = source[frmx]
                obsy = source[frmy]
            except KeyError:
                print('Unable to find key:', frm)

            data = unpack_graph(obsx, obsy)
            target[to] = data

            upd = yaml_load(extra)
            data.update(upd)

def list_get(lst, idx, default):
    return lst[idx] if idx<len(lst) else default

cmd.__tldr__ = """\
            The module recursively copies all the outputs from the source location to the target location.
            The outputs are converted to the dictionaries. Arrays, shapes, bin edges and object type are saved.
            The produced data may then be saved with `save-yaml` and `save-pickle` modules.

            Write the data from all the outputs from the 'spectra' to 'output':
            ```sh
            ./gna \\
                -- gaussianpeak --name peak --nbins 50 \\
                -- env-data -c spectra.peak output -vv \\
                -- env-print -l 40
            ```
            The last command prints the data to stdout. The value width is limited to 40 symbols.

            A common root for source and target paths may be set independently via `-s` and `-t` arguments.
            There is also a special argument `-g` to combine graphs by reading X and Y arrays from different outputs.

            Store a graph read from 'fcn.x' and 'fcn.y' as 'output.fcn_graph':
            ```sh
            ./gna \\
                -- gaussianpeak --name peak --nbins 50 \\
                -- env-data -s spectra.peak -g fcn.x fcn.y output.fcn_graph \\
                -- env-print -l 40
            ```

            Extra information may be saved with data. It should be provided as one ore more YAML dictionaries of the
            `-c` and `-g` arguments. The dictionaries will be used to update the target paths.

            Provide extra information:
            ```sh
            ./gna \\
                -- gaussianpeak --name peak --nbins 50 \\
                -- env-data -c spectra.peak output '{note: extra information}' -vv \\
                -- env-print -l 40
            ```

            See also: `env-data-root`, `save-yaml`, `save-pickle`, `save-root`.
        """

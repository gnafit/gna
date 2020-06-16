"""Plot 1d ovservables"""

from __future__ import print_function
from gna.ui import basecmd, append_typed, qualified
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from mpl_tools.helpers import savefig
import numpy as np
from tools.yaml import yaml_load
from gna.bindings import common
from gna.env import PartNotFoundError, env

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

class cmd(basecmd):
    def __init__(self, *args, **kwargs):
        basecmd.__init__(self, *args, **kwargs)

    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('-s', '--root-source', default=(), help='root namespace to copy from')
        parser.add_argument('-t', '--root-target', default=(), help='root namespace to copy to')
        parser.add_argument('-c', '--copy', nargs='+', action='append', default=[], help='Data to read and address to write', metavar=('from', 'to'))
        # parser.add_argument('-p', '--copy-plot', nargs='+', action='append', default=[], help='Data to read and address to write', metavar=('from', 'to'))

    def run(self):
        source = self.env.future.child(self.opts.root_source)
        target = self.env.future.child(self.opts.root_target)
        for copydef in self.opts.copy:
            if len(copydef)<2:
                raise Exception('Invalid number of `copy` arguments: '+str(len(copydef)))
            (frm, to), extra = copydef[:2], copydef[2:]
            try:
                obs = source[frm]
            except KeyError:
                print('Unable to find key:', frm)

            data = unpack(obs)
            target[to] = data

            for ext in extra:
                try:
                    upd = yaml_load(ext)
                except Exception:
                    print('Unable to parse extra info:', ext)
                data.update(upd)

def list_get(lst, idx, default):
    return lst[idx] if idx<len(lst) else default

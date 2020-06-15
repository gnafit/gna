"""Plot 1d ovservables"""

from gna.ui import basecmd, append_typed, qualified
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from mpl_tools.helpers import savefig
import numpy as np
from tools.yaml import yaml_load
from gna.bindings import common
from gna.env import PartNotFoundError, env
from mpl_tools.gna2mpl import is

def unpack(out):
    dtype = output.datatype()
    if dtype.kind==2:
        return unpack_hist(out, dtype)
    elif dtype.kind==1:
        return unpack_points(out, dtype)

    raise ValueError('Uninitialized output')

def unpack_hist(out, dtype):
    dtype = dtype or out.datatype()

    ret = dict(
        type  = 'hist',
        data  = out.data(),
        edges = [np.array(e) for e in dtype.edgesNd],
        shape = np.array(dtype.shape),
        ndim  = len(dtype.shape)
        )
    return ret

def unpack_points(out, dtype):
    dtype = dtype or out.datatype()

    ret = dict(
        type  = 'points',
        data  = out.data(),
        shape = np.array(dtype.shape),
        ndim  = len(dtype.shape)
        )
    return ret

class cmd(basecmd):
    def __init__(self, *args, **kwargs):
        basecmd.__init__(self, *args, **kwargs)

    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('-c', '--copy', nargs=2, action='append', default=[], help='Data to read and address to write', metavar=('from', 'to'))

    def run(self):
        ns = self.env.future
        for frm, to in self.opts.copy:
            try:
                obs = ns[frm]
            except KeyError:
                print('Unable to find key:', frm)

            data = unpack(obs)
            self.env.future[to] = data

def list_get(lst, idx, default):
    return lst[idx] if idx<len(lst) else default

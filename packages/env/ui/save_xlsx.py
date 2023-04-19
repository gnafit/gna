"""Recursively saves outputs as dictionaries with numbers and meta."""

from gna.ui import basecmd, append_typed, qualified
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from mpl_tools.helpers import savefig
import numpy as np
from gna.bindings import common
from gna.env import env
from env.lib.cwd import update_namespace_cwd

import pandas as pd
import openpyxl

def unpack(output, edges=False):
    dtype = output.datatype()
    if dtype:
        if dtype.kind==2:
            return unpack_hist(output, dtype, edges)
        elif dtype.kind==1:
            return unpack_points(output, dtype)

    raise ValueError('Uninitialized output')

def unpack_common(output):
    return output.data().copy()

def get_edges(edgeslist, mode):
    if not mode:
        return
    for edges in edgeslist:
        edges = np.asanyarray(edges)
        if mode=='+':
            yield edges[1:]
        elif mode=='-':
            yield edges[:-1]
        elif mode=='%':
            yield 0.5*(edges[1:]+edges[:-1])

def unpack_hist(output, dtype, edges_mode):
    return unpack_common(output), list(get_edges(dtype.edgesNd, edges_mode))

def unpack_points(output, dtype):
    return unpack_common(output), []

def append_points_1d(df, data, title):
    title=title or df.shape[1]
    df[title]=data

def append_hist_1d(df, data, edges, title):
    df[df.shape[1]]=edges[0]
    title=title or df.shape[1]
    df[title]=data

def append_points_2d(df, data):
    for col in data.T:
        df[df.shape[1]]=col

def append_hist_2d(df, data, edges):
    df[df.shape[1]]=edges[0]
    for e, col in zip(edges[1], data.T):
        df[e]=col

def append(df, data, edges, title):
    if len(data.shape)==1:
        if edges:
            append_hist_1d(df, data, edges, title=title)
        else:
            append_points_1d(df, data, title=title)
    elif len(data.shape)==2:
        if edges:
            append_hist_2d(df, data, edges)
        else:
            append_points_2d(df, data)
    else:
        assert False

class save_xlsx(basecmd):
    def __init__(self, *args, **kwargs):
        basecmd.__init__(self, *args, **kwargs)

    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('-o', '--output', required=True, help='output xslx file name', metavar='file.xlsx')
        parser.add_argument('-r', '--root', nargs='+', default=(), help='root namespace to copy from')
        parser.add_argument('-s', '--sheet', nargs='+', action='append', required=True, default=[], help='new sheet', metavar=('name, objects'))
        parser.add_argument('-v', '--verbose', action='count', default=0, help='verbosity')

    def run(self):
        update_namespace_cwd(self.opts, 'output')
        source = self.env.future.child(self.opts.root)

        if self.opts.verbose>1:
            print('Converting to xlsx')

        df_to_sheets = {}
        for sheetdata in self.opts.sheet:
            sheetname, datanames = sheetdata[0], sheetdata[1:]

            if not datanames:
                continue

            df = pd.DataFrame()
            df_to_sheets[sheetname]=df

            for name in datanames:
                if '=' in name:
                    title, name = name.split('=', 1)
                else:
                    title=None

                edges=False
                if name[0] in '+-%':
                    edges, name = name[0], name[1:]
                if self.opts.verbose>1:
                    print(f'  read {name}'+(edges and f'({edges})' or ''))
                obj = source[name]
                data, edges = unpack(obj, edges=edges)

                append(df, data, edges, title=title)

            if self.opts.verbose:
                print('Save output file:', self.opts.output)
            with pd.ExcelWriter(self.opts.output) as writer:
                for name, df in df_to_sheets.items():
                    df.to_excel(writer, sheet_name=name)


# save_xlsx.__tldr__ = """\
        # """

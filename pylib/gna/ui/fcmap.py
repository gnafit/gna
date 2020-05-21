# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
from gna.ui import basecmd
import numpy as np
import h5py

from gna.pointtree import PointTree

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        super(cmd, cls).initparser(parser, env)
        parser.add_argument('--fcscan', required=True)
        parser.add_argument('--output', type=str, required=True)
        parser.add_argument('--initial', type=int, default=0)
        parser.add_argument('--contexts', type=int, nargs='+', default=[1])
        parser.add_argument('--points', type=str, required=False)

    def run(self):
        fcscan = PointTree(self.opts.fcscan)

        fcmap = PointTree(self.opts.output, "w")
        fcmap.params = fcscan.params

        if self.opts.points:
            points = PointTree(self.opts.points)
        else:
            points = None

        ifield = self.opts.initial
        for path, values, ds in fcscan.iterall():
            if points and path not in points:
                continue
            chi2s = ds["chi2s"]
            print("{:20}: {} entries".format(path, len(chi2s)))
            for ctx in self.opts.contexts:
                mfield = ctx
                dchi2s = chi2s[:, ifield] - chi2s[:, mfield]
                vmap = np.sort(dchi2s)
                grp = fcmap.touch(path + "/dchi2s")
                grp.create_dataset(str(ctx), data=vmap)
                grp.attrs["entries"] = len(chi2s)
        return True

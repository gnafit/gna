from load import ROOT as R
import numpy as N
from gna.env import env, namespace
from gna import constructors as C
from mpl_tools.root2numpy import get_buffer_hist2
from gna.constructors import Histogram
from gna.configurator import NestedDict
from gna.grouping import Categories
import itertools as I

from gna.configurator import StripNestedDict, NestedDict
from tools.schema import Schema, Optional, And, Or, Use
from tools.schema import isrootfile, isreadable, haslength
from typing import Tuple, Callable, Mapping, Union, Type
from gna.bundle import TransformationBundle


class root_matrices_v01(TransformationBundle):
    """Load ROOT matrices from ROOT files v01"""

    _validator = Schema(
        And(
            {
                "bundle": object,
                "filename": And(isrootfile, isreadable),
                "names": [str],
                "formats": [str],
                Optional("labels", default=[]): [str],
                Optional("groups", default={}): {},
            },
        )
    )

    def __init__(self, *args, **kwargs):
        TransformationBundle.__init__(self, *args, **kwargs)
        self.check_nidx_dim(0, 1, "major")
        self.check_nidx_dim(0, 0, "minor")

        self.groups = Categories(self.cfg.get("groups", {}), recursive=True)
        self.vcfg: dict = self._validator.validate(StripNestedDict(self.cfg))

    @classmethod
    def _provides(cls, cfg):
        cfg: dict = cls._validator.validate(StripNestedDict(cfg))
        return (), tuple(cfg["names"])

    def build(self):
        file = R.TFile(self.vcfg["filename"], "READ")
        if file.IsZombie():
            raise Exception("Can not read ROOT file " + file.GetName())

        print("Read input file {}:".format(file.GetName()))

        for name, format, labelfmt in I.zip_longest(
            self.cfg.names, self.cfg.formats, self.cfg.get("labels", [])
        ):
            for it in self.nidx.iterate():
                if it.ndim() > 0:
                    (subst,) = it.current_values()
                else:
                    subst = ""
                hname = self.groups.format(subst, format)
                hist = file.Get(hname)
                if not hist:
                    raise Exception(f"Can not read {hname} from {file.GetName()}")

                print("  read{}: {}".format(" " + subst if subst else "", hname), end=" ")
                data = get_buffer_hist2(hist)
                print()

                if labelfmt is None:
                    labelfmt = "hist {name}\n{autoindex}"

                matrix = C.Points(data, labels=labelfmt)
                self.set_output(name, it, matrix.single())

                self.context.objects[("matrix", subst)] = hist

        file.Close()

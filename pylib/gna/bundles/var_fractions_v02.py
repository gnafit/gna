from load import ROOT as R
import numpy as N
from gna.env import env, namespace
from mpl_tools.root2numpy import get_buffer_hist1, get_bin_edges_axis
from gna.configurator import NestedDict
from gna.constructors import stdvector
from gna.bundle import TransformationBundle

class var_fractions_v02(TransformationBundle):
    def __init__(self, *args, **kwargs):
        TransformationBundle.__init__(self, *args, **kwargs)
        self.check_nidx_dim(0, 0)

    @staticmethod
    def _provides(cfg):
        return tuple((cfg.format.format(name, component=name) for name in cfg.names)), ()

    def define_variables(self):
        names_all = set(self.cfg.names)
        names_unc = list(self.cfg.fractions.keys())
        names_eval = names_all-set(names_unc)
        if len(names_eval)!=1:
            raise self.exception('User should provide N-1 fractions, the last one is not independent\n'
                                 'all: {!s}\nfractions: {!s}'.format(self.cfg.names, names_unc))
        name_eval=names_eval.pop()

        subst, names = [], ()
        for name, val in self.cfg.fractions.items():
            cname = self.cfg.format.format(name, component=name)
            names+=cname,
            par = self.reqparameter(cname, None, cfg=val, label='{} fraction'.format(name))
            subst.append(self.namespace.pathto(cname))

        label='{} fraction: '.format(name_eval)
        label+='-'.join(('1',)+names)

        name_eval = self.cfg.format.format(name_eval, component=name_eval)
        self.vd = R.VarDiff(stdvector(subst), name_eval, 1.0, ns=self.namespace)
        par=self.namespace[name_eval].get()
        par.setLabel(label)

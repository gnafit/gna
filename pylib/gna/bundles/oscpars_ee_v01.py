"""Oscillation parameters for Pee bundle v01.

Based on oscprob_v05:
    - keep only part for oscillation parameters
    - add 'fixed' option
"""

from load import ROOT as R
import gna.constructors as C
from gna.bundle import TransformationBundle

class oscpars_ee_v01(TransformationBundle):
    def __init__(self, *args, **kwargs):
        TransformationBundle.__init__(self, *args, **kwargs)
        self.check_nidx_dim(1, 1, 'both')

    @staticmethod
    def _provides(cfg):
        return ('pmns',), ()

    def build(self):
        pass

    def define_variables(self):
        pmnspars_kwargs=dict()
        pmns_name = self.get_globalname('pmns')
        ns_pmns=self.namespace(pmns_name)

        #
        # Define oscillation parameters
        #
        pars = self.cfg['parameters']
        otherpars = dict(
                SinSq23 = 0.542,
                )
        labels = dict([
            ('DeltaMSq13', 'Mass splitting |Δm²₁₃|'),
            ('DeltaMSq23', 'Mass splitting |Δm²₂₃|'),
            ('DeltaMSq12', 'Mass splitting |Δm²₂₁|'),
            ('SinSqDouble13', 'Reactor mixing amplitude sin²2θ₁₃ '),
            ('SinSqDouble12', 'Solar mixing amplitude sin²2θ₁₂'),
            ('SinSq23', 'Atmospheric mixing angle sin²θ₂₃'),
            ])

        missing=-1
        allfixed = self.cfg.get('fixed', False)
        for name, label in labels.items():
            if name in pars:
                central, sigma = pars[name], None
                free = not allfixed
            elif name in ('DeltaMSq13', 'DeltaMSq23'):
                missing+=1
                continue
            else:
                try:
                    central, sigma = otherpars[name], None
                except KeyError:
                    raise self.exception(f'Oscillation parameter {name} is not initialized')
                free = False
            if isinstance(central, (tuple, list)):
                central, sigma=central

            if sigma:
                ns_pmns.reqparameter(name, central=central, sigma=sigma, fixed=allfixed, label=label)
            else:
                ns_pmns.reqparameter(name, central=central, free=free, fixed=not free, label=label)

        if missing:
            raise self.exception('Either DeltaMSq13 or DeltaMSq23 should be initialized')

        ns_pmns.reqparameter('Alpha', type='discrete', default='normal', variants={'normal': 1.0, 'inverted': -1.0}, label='Neutrino mass ordering α')
        ns_pmns.reqparameter('Delta', type='uniformangle', central=0.0, fixed=True, label='CP violation phase δ(CP)')

        #
        # Define oscillation expressions to provide missing and conjucated oscillation parameters
        # Define PMNS oscillation parameters
        #
        with ns_pmns:
            self._expressions_pars=C.OscillationExpressions(ns=ns_pmns)
            self._expressions_pmns=C.PMNSExpressionsC(ns=ns_pmns)

        #
        # Define oscillation weights
        #
        names = C.stdvector(['comp0', 'comp12', 'comp13', 'comp23'])
        with ns_pmns:
            self._expressions_oscprob=R.OscProbPMNSExpressions(R.Neutrino.ae(), R.Neutrino.ae(), names, ns=ns_pmns)
            ns_pmns.materializeexpressions()

        for i, vname in enumerate(names):
            ns_pmns[vname].setLabel('Psur(ee) weight %i: %s '%(i, vname))

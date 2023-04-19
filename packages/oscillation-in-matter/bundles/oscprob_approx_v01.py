"""Oscillation probability in matter (approximate) bundle v01.

Based on oscprob_v05
"""

from load import ROOT as R
import gna.constructors as C
from gna.bundle import TransformationBundle

class oscprob_approx_v01(TransformationBundle):
    def __init__(self, *args, **kwargs):
        TransformationBundle.__init__(self, *args, **kwargs)
        self.check_nidx_dim(2, 2, 'major')

        try:
            source_name, detector_name = self.cfg.bundle.major
        except:
            raise Exception('Unable to obtain major indices: source, detector and OP component')
        self.idx_source = self.nidx_major.get_subset(source_name)
        self.idx_detector = self.nidx_major.get_subset(detector_name)

    @staticmethod
    def _provides(cfg):
        return ('pmns',), ('oscprob_msw_approx',)

    def build(self):
        pmns_name = self.get_globalname('pmns')
        for it_source in self.idx_source:
            for it_detector in self.idx_detector:
                it_dist = it_source+it_detector
                dist = it_dist.current_format(name='baseline')

                oscprobkey = it_dist.current_format('{autoindex}')[1:]

                with self.namespace, self.namespace(pmns_name):
                    for it_minor in self.nidx_minor:
                        oscprob = self.context.objects[(pmns_name,oscprobkey)] = R.OscProb3ApproxMSW(dist)

                        it = it_source+it_detector+it_minor

                        trans = oscprob.oscprob
                        if self.nidx_minor:
                            trans.setLabel( it.current_format('Approximate MSW OP: |{reactor}-\\>{detector}|'+it_minor.current_format()) )
                        else:
                            trans.setLabel( it.current_format('Approximate MSW OP: |{reactor}-\\>{detector}') )
                        output = trans.oscprob
                        enu_input  = trans['Enu']

                        self.set_input('oscprob_msw_approx',  it, enu_input, argument_number=0)
                        self.set_output('oscprob_msw_approx', it, output)

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
        density = self.cfg.get('density', None)

        labels = dict([
            ('DeltaMSq13', 'Mass splitting |Δm²₁₃|'),
            ('DeltaMSq23', 'Mass splitting |Δm²₂₃|'),
            ('DeltaMSq12', 'Mass splitting |Δm²₂₁|'),
            ('SinSqDouble13', 'Reactor mixing amplitude sin²2θ₁₃ '),
            ('SinSqDouble12', 'Solar mixing amplitude sin²2θ₁₂'),
            ('SinSq23', 'Atmospheric mixing angle sin²θ₂₃'),
            ])

        missing=-1
        for name, label in labels.items():
            if name in pars:
                central, sigma = pars[name], None
                free = True
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
                ns_pmns.reqparameter(name, central=central, sigma=sigma, label=label)
            else:
                ns_pmns.reqparameter(name, central=central, free=free, fixed=not free, label=label)

        if missing:
            raise self.exception('Either DeltaMSq13 or DeltaMSq23 should be initialized')

        ns_pmns.reqparameter('Alpha', type='discrete', default='normal', variants={'normal': 1.0, 'inverted': -1.0}, label='Neutrino mass ordering α')
        ns_pmns.reqparameter('Delta', type='uniformangle', central=0.0, fixed=True, label='CP violation phase δ(CP)')

        label_rho = 'Earth crust density in g/cm^3'
        if density is None:
            self.namespace.reqparameter("rho", central=2.6, fixed=True, label=label_rho)
        else:
            self.namespace.reqparameter("rho", central=density, fixed=True, label=label_rho)
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

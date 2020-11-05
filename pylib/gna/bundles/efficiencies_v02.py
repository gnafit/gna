
from load import ROOT as R
import numpy as N
import gna.constructors as C
from gna.bundle import TransformationBundle
from collections import OrderedDict

class efficiencies_v02(TransformationBundle):
    mode = 'correlated' # 'uncorrelated'
    def __init__(self, *args, **kwargs):
        TransformationBundle.__init__(self, *args, **kwargs)
        self.check_nidx_dim(0, 0, 'major')

        self.init_data()

    @staticmethod
    def _provides(cfg):
        return ('eff', 'effunc_corr', 'effunc_uncorr', 'norm'), ()

    def init_data(self):
        from gna.configurator import configurator

        efficiencies = self.cfg.efficiencies
        if isinstance(efficiencies, str):
            try:
                data = configurator(self.cfg.efficiencies)
            except:
                raise Exception('Unable to open or parse file: '%self.cfg.efficiencies )

            try:
                efficiencies = data.efficiencies
                self.mode    = data.mode
            except:
                raise Exception('Efficiencies file (%s) should contain "efficiencies" and "mode" field'%self.cfg.efficiencies)
        else:
            self.mode = self.cfg.mode

        try:
            self.efficiencies = N.array( [tuple(e) for e in efficiencies], dtype=[('name', 'S', 20), ('eff', 'd'), ('corr', 'd'), ('uncorr', 'd')] )
        except:
            print('Unable to parse efficiencies:')
            print(efficiencies)
            raise

        if not self.mode in ['absolute', 'relative']:
            raise Exception( 'Unsupported uncertainty type: %s (should be "absolute" or "relative")'%self.mode )

    def define_variables(self):
        from gna.configurator import uncertaindict

        eff=self.efficiencies['eff']
        if self.mode=='relative':
            relunc_uncorr = self.efficiencies['uncorr']
            relunc_corr   = self.efficiencies['corr']
        else:
            relunc_uncorr = self.efficiencies['uncorr']/eff
            relunc_corr   = self.efficiencies['corr']/eff

        eff_tot = eff.prod()
        relunc_uncorr_tot = (relunc_uncorr**2).sum()**0.5
        relunc_corr_tot = (relunc_corr**2).sum()**0.5

        self.reqparameter('eff', None, central=eff_tot, fixed=True, label='absolute efficiency')

        if self.cfg.get('norm'):
            self.reqparameter('norm', None, central=1.0, free=True, label='global normalization' )

        if self.cfg.get('correlated'):
            self.reqparameter('effunc_corr', None, central=1.0, sigma=relunc_corr_tot, label='correlated efficiency uncertainty (relative)')

        if self.cfg.get('uncorrelated'):
            for i, it in enumerate(self.nidx_minor.iterate()):
                self.reqparameter('effunc_uncorr', it, central=1.0, sigma=relunc_uncorr_tot, label='Uncorrelated efficiency uncertainty (relative)'  )

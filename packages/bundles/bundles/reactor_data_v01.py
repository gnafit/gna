from load import ROOT as R
import numpy as np
import gna.constructors as C
from gna.bundle import *
from gna.configurator import StripNestedDict
from tools.schema import *

Units = R.NeutrinoUnits

conversion_distance = { 'meter': 1.e-3, 'kilometer': 1.0, }
conversion_power = { 'megawatt': 1.e-3, 'gigawatt': 1.0 }
conversion_distance['m']=conversion_distance['meter']
conversion_distance['km']=conversion_distance['kilometer']
conversion_power['mw']=conversion_power['megawatt']
conversion_power['gw']=conversion_power['gigawatt']

datatypes = {
        'name': 'U25',
        'power': 'd',
        'distance': 'd',
        'duty_cycle': 'd',
        }

labels = {
        'distance':    'km, baseline between {detector} and {reactor}',
        'weight':      'cm⁻², 1/(4πL²) for {detector} and {reactor}',
        'power':       'GW, thermal power for {reactor}',
        'power_scale': 'thermal power scale for {reactor}',
        'duty_cycle': 'duty_cycle/UCF for {reactor}',
}

class reactor_data_v01(TransformationBundle):
    """Load reactor data (v01)

    Loads baselines, thermal power and duty cycle from a text table

    Indices:
        - major - reactor
        - minor - replicas (for baseline and baselineweight)

    Based on a reactor_baselines_v03.
    """
    def __init__(self, *args, **kwargs):
        TransformationBundle.__init__(self, *args, **kwargs)
        self.check_nidx_dim(1, 1, 'major')

        self.vcfg = self._validator.validate(StripNestedDict(self.cfg))
        self.init_data()

    _validator = Schema({
        'bundle': object,
        'filename': And(str, isreadable, Or(isfilewithext('dat'), isfilewithext('txt'), isfilewithext('tsv'))),
        'units': {
            'power': Or(*conversion_power.keys()),
            'distance': Or(*conversion_distance.keys())
            },
        'columns': And([Or('name', 'power', 'distance', 'duty_cycle')], haslength(exactly=4)),
        Optional('labels', default=labels): {Optional(key, default=label): str for key, label in labels.items()},
        Optional('relative_uncertainty', default=None): float,
        })

    @staticmethod
    def _provides(cfg):
        return ('baseline', 'baselineweight', 'thermal_power_nominal', 'thermal_power_scale', 'duty_cycle'), ()

    def init_data(self):
        '''Read configurations of reactors and detectors from either files or dicts'''
        dtype = [(name, datatypes[name]) for name in self.vcfg['columns']]
        self.data=np.loadtxt(self.vcfg['filename'], dtype=dtype)

        self.data['power']*=conversion_power[self.vcfg['units']['power']]
        self.data['distance']*=conversion_distance[self.vcfg['units']['distance']]

    def define_variables(self):
        labels = self.vcfg['labels']

        relunc=self.vcfg['relative_uncertainty']
        fixed=relunc is None

        const = 0.25/np.pi*1.e-10 # Convert 1/km2 to 1/cm2
        names = self.data['name'].tolist()
        for idx_reactor in self.nidx_major:
            reactor_name, = idx_reactor.current_values()

            try:
                nameidx = names.index(reactor_name)
            except ValueError as e:
                raise self.exception(f'Data not provided for the reactor {reactor_name}')

            data = self.data[nameidx]
            distance, power = data['distance'], data['power']
            duty_cycle = data['duty_cycle']

            self.reqparameter('thermal_power_nominal', idx_reactor, central=power, fixed=True,
                              label=idx_reactor.current_format(labels['power']))

            self.reqparameter('thermal_power_scale', idx_reactor, central=1.0, relsigma=relunc, fixed=fixed,
                              label=idx_reactor.current_format(labels['power_scale']))

            self.reqparameter('duty_cycle', idx_reactor, central=duty_cycle, fixed=True,
                              label=idx_reactor.current_format(labels['duty_cycle']))

            for idx_other in self.nidx_minor:
                idx = idx_reactor + idx_other

                self.reqparameter('baseline', idx, central=distance, fixed=True,
                                  label=idx.current_format(labels['distance']))

                self.reqparameter('baselineweight', idx, central=const/distance**2, fixed=True,
                                  label=idx.current_format(labels['weight']))


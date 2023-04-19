from load import ROOT as R
import gna.constructors as C
import numpy as np
from gna.bundle import *
from gna.env import env
from tools.root_helpers import TFileContext
from mpl_tools.root2numpy import get_buffers_graph_or_hist1
from tools.data_load import read_object_auto

class reactor_snf_spectra_v06(TransformationBundle):
    """A bundle for spent nuclear fuel contribution v06
    Changes since v05:
        - enable minor indexing
    """

    par_name = 'snf_scale'
    def __init__(self, *args, **kwargs):
        TransformationBundle.__init__(self, *args, **kwargs)
        self.check_nidx_dim(1, 1, 'major')

        self.snf_raw_data = dict()
        self._load_data()

    @staticmethod
    def _provides(cfg):
            return ('snf_scale',), ('snf_correction',)

    def _load_datum(self, filename):
        return read_object_auto(filename, name=self.cfg.get('objectname'), verbose=True, suffix=' [SNF]')

    def _load_data(self):
        """Read raw input spectra. Can be either average spectrum or individual
        for each reactor. """
        def _load_average(datapath):
            self.snf_raw_data['average'] = self._load_datum(datapath)

        def _load_by_reactor(template):
            for reac_idx in self.nidx.get_subset('r'):
                reac, = reac_idx.current_values()
                key = reac_idx.names()[0]
                datapath = template.format(**{key: reac})
                self.snf_raw_data[reac] = self._load_datum(datapath)

        try:
            datapath = self.cfg.snf_average_spectra
            _load_average(datapath)
        except KeyError:
            try:
                data_template = self.cfg.snf_pools_spectra
                _load_by_reactor(data_template)
            except KeyError:
                raise

    def build(self):
        self.objects = {}

        snf_energies_in = self.objects['energies_in'] = {}
        snf_values_in   = self.objects['values_in'] = {}
        for idxmajor in self.nidx_major:
            reac, = idxmajor.current_values()

            try:
                x, y = self.snf_raw_data[reac]
            except KeyError:
                x, y = self.snf_raw_data['average']

            snf_energies_in[reac] = C.Points(x, labels=f'SNF {reac}: input energy')
            snf_values_in[reac]   = C.Points(y, labels=f'SNF {reac}: input values')


        snapshots = self.objects['snapshots'] = []
        products = self.objects['products'] = []
        sums = self.objects['sums'] = []
        for idxmajor in self.nidx_major:
            reac, = idxmajor.current_values()

            weights = [idxmajor.current_format(name=self.par_name)]
            for idxminor in self.nidx_minor:
                idx = idxminor+idxmajor

                snf_spectra = C.InterpLinear()
                snf_spectra.set_overflow_strategy(R.GNA.Interpolation.Strategy.Constant)
                snf_spectra.set_underflow_strategy(R.GNA.Interpolation.Strategy.Constant)

                insegment = snf_spectra.transformations.front()
                insegment.setLabel(idx.current_format("{{Segments|{autoindex}}}"))

                interpolator_trans = snf_spectra.transformations.back()
                interpolator_trans.setLabel(idx.current_format("{{SNF interpolated|{autoindex}}}"))

                _snf_energy = snf_energies_in[reac]
                _snf_value = snf_values_in[reac]

                _snf_energy >> (insegment.edges, interpolator_trans.x)
                _snf_value  >> interpolator_trans.y

                snap = C.Snapshot(labels=idxminor.current_format('{{Nominal spectra for SNF from {reac}|{autoindex}}}', reac=reac))
                snapshots+=snap,

                self.set_input('snf_correction', idx, (insegment.points, interpolator_trans.newx), argument_number=0)
                self.set_input('snf_correction', idx, snap.single_input(), argument_number=1)
                product = C.Product(outputs=[snap.single(), interpolator_trans.single()],
                                    labels=idxminor.current_format('{{Nominal SNF contribution for {reac}|{autoindex}}}',
                                                                  reac=reac)
                                    )
                products+=product,

                outputs = [product.single()]
                with self.namespace:
                    final_sum = C.WeightedSum(weights, outputs,
                                              labels=idxminor.current_format('SNF contribution for {reac}|{autoindex}', reac=reac))
                sums+=final_sum,
                self.set_output("snf_correction", idx, final_sum.single())

    def define_variables(self):
        for idxmajor in self.nidx_major:
            reac, = idxmajor.current_values()
            self.reqparameter(self.par_name, idxmajor, central=1., relsigma=1,
                              label=f"SNF norm for reactor {reac}")

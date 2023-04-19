from load import ROOT as R
import gna.constructors as C
import numpy as np
from gna.bundle import *
from gna.env import env
from tools.data_load import read_object_auto

class reactor_offeq_spectra_v05(TransformationBundle):
    """Reactor spectra offequilibrium correction
    Applicable to ILL based 235U/239Pu/241Pu spectra (Huber, Schreckenbach)

    Indexing:
      major - gets its own nuisance parameter
      minor - gets the correction

      major index may or may not contain the isotope index

    Chages since v04:
    - add ROOT input support
    """
    _unity_name = 'offequilibrium_unity'
    _par_name = "offeq_scale"
    def __init__(self, *args, **kwargs):
        TransformationBundle.__init__(self, *args, **kwargs)
        self.check_nidx_dim(0, 2, 'major')
        self.offeq_raw_spectra = dict()

        try:
            self.nidx_iso, self.nidx_noniso = self.nidx.split(self.cfg.get('isotope_index', 'i'))
        except IndexError:
            self.nidx_iso, self.nidx_noniso = self.nidx.split([])

        self._load_data()

    @staticmethod
    def _provides(cfg):
            return ('offeq_scale',), ('offeq_correction',)

    def _load_data(self):
        """Read raw input spectra"""
        data_template = self.cfg.offeq_data
        objname_template = self.cfg.get('objectnamefmt', '')
        for isotope in self.nidx_iso:
            if isotope.ndim():
                iso_name, = isotope.current_values()
            else:
                iso_name = ''

            datapath = data_template.format(isotope=iso_name)
            objectname = objname_template.format(isotope=iso_name)
            try:
                self.offeq_raw_spectra[iso_name] = \
                        read_object_auto(datapath, name=objectname, verbose=True, suffix=' [{} offeq]'.format(iso_name))
            except (IOError, ValueError):
                # U238 doesn't have offequilibrium correction
                if iso_name == 'U238':
                    pass
                else:
                    raise
            assert len(self.offeq_raw_spectra) != 0, "No data loaded"

    def build(self):
        for idx_iso in self.nidx_iso:
            try:
                iso, = idx_iso.current_values()
            except ValueError:
                iso = ''
            name = idx_iso.current_format(name="offeq_correction")

            try:
                _offeq_energy, _offeq_spectra = map(C.Points, self.offeq_raw_spectra[iso])
                _offeq_energy.points.setLabel("Original energies for offeq spectrum of {}".format(iso))
                dummy = False
            except KeyError:
                # U238 doesn't have offequilibrium correction so just pass 1.
                if iso != 'U238':
                    raise
                dummy = True

            if not dummy:
                offeq_spectra = C.InterpLinear(labels=idx_iso.current_format('Correction for {autoindex} spectra'))
                offeq_spectra.set_overflow_strategy(R.GNA.Interpolation.Strategy.Constant)
                offeq_spectra.set_underflow_strategy(R.GNA.Interpolation.Strategy.Constant)

                insegment = offeq_spectra.transformations.front()
                insegment.setLabel("Offequilibrium segments")

                interpolator_trans = offeq_spectra.transformations.back()
                interpolator_trans.setLabel(idx_iso.current_format("Interpolated spectral correction for {autoindex}"))

                _offeq_energy >> (insegment.edges, interpolator_trans.x)
                _offeq_spectra >> interpolator_trans.y

            first = True
            for idx_noniso in self.nidx_noniso:
                idx = idx_iso + idx_noniso
                idx_major = idx.get_subset(self.nidx_major)

                if dummy:
                    dummy = C.Identity() #just to serve 1 input
                    self.set_input('offeq_correction', idx, dummy.single_input(), argument_number=0)
                    self.set_input('offeq_correction', idx, passthrough.single_input(), argument_number=1)
                    self.set_output("offeq_correction", idx, passthrough.single())
                    continue

                if first:
                    # Enu
                    self.set_input('offeq_correction', idx, (insegment.points, interpolator_trans.newx), argument_number=0)
                else:
                    self.set_input('offeq_correction', idx, (), argument_number=0)

                passthrough = C.Identity(labels=idx.current_format('Nominal spectrum at {autoindex}'))

                # Anue spectra
                self.set_input('offeq_correction', idx, (passthrough.single_input()), argument_number=1)

                snap = C.Snapshot(passthrough.single(), labels=idx.current_format('Snapshot {autoindex} spectra'))

                prod = C.Product(labels=idx.current_format('Initial {autoindex} spectrum x offequilibrium corr'))
                prod.multiply(interpolator_trans.single())
                prod.multiply(snap.single())

                outputs = [passthrough.single(), prod.single()]
                weights = [self._unity_name, idx_major.current_format(name=self._par_name)]

                with self.namespace:
                    label = idx.current_format('{{anue spectrum, offeq corrected|{autoindex}}}')
                    final_sum = C.WeightedSum(weights, outputs, labels=label)

                self.context.objects[name] = final_sum
                self.set_output("offeq_correction", idx, final_sum.single())

    def define_variables(self):
        self.reqparameter(self._unity_name, None, central=1, fixed=True, label="Unitary offequilibrium weight")
        relsigma = self.cfg['relsigma']
        for idx in self.nidx_major:
            self.reqparameter(self._par_name, idx, central=1., relsigma=relsigma, label="Offequilibrium norm for {autoindex}")


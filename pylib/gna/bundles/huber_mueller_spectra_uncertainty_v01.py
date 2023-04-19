import os
import numpy as np
from collections import defaultdict
from load import ROOT as R
import gna.constructors as C
from gna.bundle import TransformationBundle

class huber_mueller_spectra_uncertainty_v01(TransformationBundle):
    '''Provides uncorrelated and correlated spectral uncertainties for
    reactor antineutrino spectrum in Huber-Mueller model.

    Parameters for configuration:
        files_uncorr -- [pathes] to files with uncorrelated uncertainties
        files_corr -- [pathes] to files with correlated uncertainties
        iso_idx -- name of isotope index
        reac_idx -- name of reactor index
        ns_name -- namespace where to store pars
    '''

    def __init__(self, *args, **kwargs):
        TransformationBundle.__init__(self, *args, **kwargs)
        self.check_nidx_dim(2, 2, 'major')
        self.check_nidx_dim(0, 0, 'minor')
        self.unc_ns = self.namespace(self.cfg.ns_name)

        self.uncorrelated_vars = defaultdict(dict)
        self.correlated_vars = defaultdict(dict)
        self.total_unc = defaultdict(dict)
        self.objects = defaultdict(dict)

        self._dtype = [('enu', 'd'), ('unc', 'd')]
        self.load_data()
        self._enu_to_bins()

    @staticmethod
    def _provides(cfg):
        return (), ('corrected_spectrum',)

    def build(self):
        '''Bringing uncertainties together with spectrum.
        Spectrum values are grouped by bins in antineutrino
        energy (binning is read together with uncertainties).
        Each bin is corrected accordingly and corrected spectrum is provided
        as output.
        '''
        order = ','.join((self.cfg.iso_idx, self.cfg.reac_idx))
        reversed_order = ','.join((self.cfg.reac_idx, self.cfg.iso_idx))
        order_from_idx =self.nidx_major.comma_list()
        if order_from_idx == order:
            normal_order = True
        elif order_from_idx == reversed_order:
            normal_order = False
        else:
            raise ValueError("Indicies from config and real don't match")

        for idx in self.nidx_major:
            if normal_order:
                iso, reac, = idx.current_values()
            else:
                reac, iso, = reac_idx.current_values()

            unc = self.total_unc[reac][iso]
            nominal = R.FillLike(1., labels="Nominal weight for spectrum")
            unc.single() >> nominal.single_input()
            total = C.Sum(outputs=[unc, nominal],
                          labels=f"Nominal weigts + correction for {iso} in {reac}")

            correction = R.ReactorSpectrumUncertainty(self.binning.single(), total.single())
            correction.insegment.setLabel(f"Segments in HM-bins for {iso} in {reac}")
            correction.transformations.in_bin_product.setLabel(f"Correction to antineutrino spectrum in HM model for {iso} in {reac}")

            self.set_input("corrected_spectrum", idx, correction.insegment.points, argument_number=0)
            self.set_input("corrected_spectrum", idx, correction.transformations.in_bin_product.spectrum, argument_number=1)
            self.set_output("corrected_spectrum", idx, correction.transformations.in_bin_product.corrected_spectrum)




    def define_variables(self):
        """Define variables for correlated and uncorrelated uncertainties.
        Correlated uncertainties are represented as WeightedSums of fixed
        uncertainties from HM model multiplied by single weight for each reactor and shared
        for isotopes within same reactor.

        Uncorrelated uncertainties are represented as VarArrays of uncertain
        parameters. Each parameter uncertainty corresponds to uncorrelated
        uncertainty from HM model multiplied for given isotope in given bin.
        All parameters are uncorrelated between reactors and isotopes.
        """
        num_of_bins = len(self.bins) - 1
        with self.unc_ns:
            for reac_idx in self.nidx_major.get_subset(self.cfg.reac_idx):
                reac, = reac_idx.current_values()
                corr_name = "corr_unc." + reac_idx.current_format()
                self.unc_ns.reqparameter(corr_name, central=0., sigma=1.,
                        label=f"Correlated uncertainty in {reac}")
                for iso_idx in self.nidx_major.get_subset(self.cfg.iso_idx):
                    iso, = iso_idx.current_values()
                    idx = iso_idx + reac_idx
                    uncorr_temp = "uncorr_unc." + idx.current_format() + ".{bin}"
                    vars = []
                    for bin, iso_unc in zip(range(num_of_bins), self.uncertainties['uncorr'][iso]):
                        name = uncorr_temp.format(bin=bin)
                        vars.append(name)
                        self.unc_ns.reqparameter(name, central=0., sigma=iso_unc,
                                label=f'''Uncorrelated uncertainty for {iso} in {reac}, bin {bin}''')
                    uncorr_unc = C.VarArray(vars, ns=self.unc_ns,
                                            labels=f'Uncorrelated uncertainties for {iso} in {reac}')
                    self.uncorrelated_vars[reac][iso] = uncorr_unc

                    tmp = C.Points(self.uncertainties['corr'][iso],
                            labels=f'Fixed array of correlated uncertainties for {iso} in {reac}')
                    corr_unc = C.WeightedSum([corr_name], [tmp], ns=self.unc_ns,
                                              labels=f"Correlated uncertainties for {iso} in {reac}")
                    self.correlated_vars[reac][iso] = corr_unc
                    self.total_unc[reac][iso] = C.Sum([uncorr_unc, corr_unc])

    def load_data(self):
        """Read correlated and uncorrelated uncertainties for each isotope"""
        def _get_file(templates):
            '''Get path to file from templates if it exists in filesystem'''
            for temp in templates:
                attempt = temp.format(isotope=isotope)
                if os.path.exists(attempt):
                    return attempt
            raise KeyError(f"No file matching f{templates} for {isotope}!")

        self.uncertainties = defaultdict(dict)
        for it in self.nidx_major.get_subset(self.cfg.iso_idx):
            isotope, = it.current_values()
            uncorr_file = _get_file(self.cfg.files_uncorr)
            self.uncertainties['uncorr'][isotope] = self.load_file(uncorr_file)

            corr_file = _get_file(self.cfg.files_corr)
            self.uncertainties['corr'][isotope] = self.load_file(corr_file)

    def load_file(self, fname):
        try:
            enu, unc = np.loadtxt(fname, self._dtype, unpack=True)
            # check for consistent binning in energy for different isotopes
            # and corr/uncorrelated errors
            try:
                assert np.allclose(self._enu, enu), "Inconsistent binning in energy isotopes"
            except AttributeError:
                self._enu = enu
        except:
            raise IOError(f"Failed to read uncertainties from {fname}")
        else:
            return unc

    def _enu_to_bins(self):
        """Compute bin edges from energy points. Bins are checked to have the
        same width"""
        enu = self._enu
        bin_width = enu[1] - enu[0]
        widthes = enu[1:] - enu[:-1]
        assert np.allclose(widthes, bin_width), "Bins have different widthes!"
        bins = self._enu - bin_width/2.
        self.bins = np.append(bins, enu[-1]+bin_width/2.) # get rightmost bin edge
        self.binning = C.Points(self.bins, labels='Bins for HM corrections')

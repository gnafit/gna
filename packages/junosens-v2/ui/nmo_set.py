# encoding: utf-8
"""Switch neutrino mass ordering while keeping particular mass splitting the same"""

from __future__ import print_function
from gna.ui import basecmd
from load import ROOT as R
import numpy as np

class cmd(basecmd):
    def __init__(self, *args):
        basecmd.__init__(self, *args)
        self.toggle_keep={
                'DeltaMSq23': {'DeltaMSq23': self._toggle_alpha,      'DeltaMSqEE':  self._change_23_keep_ee,
                               'DeltaMSq13': self._change_23_keep_31, 'DeltaMSqAVG': self._change_23_keep_avg},
                'DeltaMSq13': {'DeltaMSq13': self._toggle_alpha,      'DeltaMSqEE':  self._change_13_keep_ee},
                'DeltaMSqEE': {'DeltaMSqEE': self._toggle_alpha},
                }

    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('pmns', help='namespace with pmns')
        parser.add_argument('-t', '--toggle', action='store_true', help='toggle ordering')
        parser.add_argument('-k', '--keep-splitting', default='ee', choices=('23', '13', 'ee', 'avg'), help='the mass splitting to keep while switching ordering')
        parser.add_argument('-v', '--verbose', help='verbosity level')


    def init(self):
        self.opts.keep_splitting = 'DeltaMSq'+self.opts.keep_splitting.upper()

        self._getpars()
        self._determineparleading()
        self._determineparfixed()

        if self.opts.toggle:
            self._value1 = self.parkeptvalue()
            self.toggle()
            self._value2 = self.parkeptvalue()
            self._check()

    def toggle(self):
        leading = self.parleading.name()
        kept = self.opts.keep_splitting

        method = self.toggle_keep.get(leading, {}).get(kept)
        if not method:
            raise Exception('Unable to switch hierarchy, change {} and keep {}'.format(leading, kept))

        method()

    def _toggle_alpha(self):
        alpha = self.alpha.value()

        if alpha=='inverted':
            self.alpha.set('normal')
            return 1.0

        self.alpha.set('inverted')
        return -1

    def _change_23_keep_ee(self):
        u"""
            |dm²₃₂|(IO) = |dm²₃₂|(NO) +    2 cos²₁₂|dm²₂₁|
            |dm²₃₂|(NO) = |dm²₃₂|(IO) -    2 cos²₁₂|dm²₂₁|
            |dm²₃₂|₂    = |dm²₃₂|₁    + α₁ 2 cos²₁₂|dm²₂₁|
        """
        v1     = self.parleading.value()
        alpha1 = self.alpha._variable.value()
        self._toggle_alpha()
        self.parleading.set(v1 + 2 * alpha1 * self.cosSq12.value() * self.dm12.value())

    def _change_23_keep_avg(self):
        u"""
            |dm²₃₂|(IO) = |dm²₃₂|(NO) +   |dm²₂₁|
            |dm²₃₂|(NO) = |dm²₃₂|(IO) -   |dm²₂₁|
            |dm²₃₂|₂    = |dm²₃₂|₁    + α₁|dm²₂₁|
        """
        v1     = self.parleading.value()
        alpha1 = self.alpha._variable.value()
        self._toggle_alpha()
        self.parleading.set(v1 + alpha1 * self.dm12.value())

    def _change_23_keep_31(self):
        u"""
            |dm²₃₂|(IO) = |dm²₃₂|(NO) +    2 |dm²₂₁|
            |dm²₃₂|(NO) = |dm²₃₂|(IO) -    2 |dm²₂₁|
            |dm²₃₂|₂    = |dm²₃₂|₁    + α₁ 2 |dm²₂₁|
        """
        v1     = self.parleading.value()
        alpha1 = self.alpha._variable.value()
        self._toggle_alpha()
        self.parleading.set(v1 + 2 * alpha1 * self.dm12.value())

    def _change_13_keep_ee(self):
        u"""
            |dm²₃₁|(IO) = |dm²₃₁|(NO) -    2 sin²₁₂|dm²₂₁|
            |dm²₃₁|(NO) = |dm²₃₁|(IO) +    2 sin²₁₂|dm²₂₁|
            |dm²₃₁|₂    = |dm²₃₁|₁    - α₁ 2 sin²₁₂|dm²₂₁|
        """
        v1     = self.parleading.value()
        alpha1 = self.alpha._variable.value()
        self._toggle_alpha()
        self.parleading.set(v1 - 2 * self.sinSq12.value() * self.dm12.value())

    def _check(self):
        if not np.isclose(self._value1, self._value2, atol=0, rtol=1.e-8):
            raise Exception('Mass splitting value changed too much')

    def _getpars(self):
        pmns = self.pmns = self.env.globalns(self.opts.pmns)
        try:
            self.alpha   = pmns['Alpha']
            self.dm12    = pmns['DeltaMSq12']
            self.dm13    = pmns['DeltaMSq13']
            self.dm23    = pmns['DeltaMSq23']
            self.dmEE    = pmns['DeltaMSqEE']
            self.sinSq12 = pmns['SinSq12']
            self.cosSq12 = pmns['CosSq12']
        except KeyError:
            raise Exception('Unable to retrieve one of the oscillation parameters')

    def _determineparleading(self):
        ParType = R.GaussianParameter('double')

        for par in (self.dm23, self.dmEE, self.dm13):
            if isinstance(par, ParType):
                break
        else:
            raise Exception('Unable to determine leading parameter')

        self.parleading = par

    def _determineparfixed(self):
        if self.opts.keep_splitting=='DeltaMSqAVG':
            self.parkeptvalue=lambda: 0.5*(self.dm13.value()+self.dm23.value())
        else:
            for par in (self.dm23, self.dmEE, self.dm13):
                if par.name()==self.opts.keep_splitting:
                    self.parkeptvalue = par.value
                    break
            else:
                raise Exception('Unable to determine fixed parameter')

"""Usage example:
   ./gna \
    -- mpl -r 'lines.linewidth: 1.4' 'axes.formatter.limits: [-2, 2]' \
    -- exp --ns juno juno_sensitivity_v03_common -vv --energy-model eres --eres-sigma 0.03 \
    -- env-print spectra.juno -l 40 \
    -- mpl --figure 'figsize: [8, 4]' \
    -- spectrum \
                -p juno.AD1.eres \
                -l 'NO' \
                --plot-type hist --scale \
                --plot-kwargs '{color: red, linestyle: dashed, alpha: 0.5}' \
    -- ns --ns juno.pmns --print \
    -- nmo-set juno.pmns -t -k 23 \
    -- ns --ns juno.pmns --print \
    -- spectrum \
                -p juno.AD1.eres \
                -l 'IO: same $\\Delta m^2_{32}$' \
                --plot-type hist --scale \
                --plot-kwargs '{color: blue}' \
    -- ns --ns juno.pmns --print \
    -- nmo-set juno.pmns -t -k 23\
    -- ns --ns juno.pmns --print \
    -- spectrum \
                -p juno.AD1.eres \
                -l 'NO: revert' \
                --plot-type hist --scale \
                --plot-kwargs '{color: magenta, linestyle: dotted, alpha: 0.5}' \
    -- ns --ns juno.pmns --print \
    -- nmo-set juno.pmns -t -k ee \
    -- ns --ns juno.pmns --print \
    -- spectrum \
                -p juno.AD1.eres \
                -l 'IO: same $\\Delta m^2_{\mathrm{ee}}$' \
                --plot-type hist --scale \
                --plot-kwargs '{color: green}' \
    -- nmo-set juno.pmns -t -k ee \
    -- ns --ns juno.pmns --print \
    -- nmo-set juno.pmns -t -k 13 \
    -- ns --ns juno.pmns --print \
    -- spectrum \
                -p juno.AD1.eres \
                -l 'IO: same $\\Delta m^2_{\mathrm{31}}$' \
                --plot-type hist --scale \
                --plot-kwargs '{color: gold}' \
    -- nmo-set juno.pmns -t -k 13 \
    -- ns --ns juno.pmns --print \
    -- nmo-set juno.pmns -t -k avg \
    -- ns --ns juno.pmns --print \
    -- spectrum \
                -p juno.AD1.eres \
                -l 'IO: same $\\Delta m^2_{\mathrm{avg}}$' \
                --plot-type hist --scale \
                --plot-kwargs '{color: cyan}' \
    -- nmo-set juno.pmns -t -k avg \
    -- ns --ns juno.pmns --print \
    -- mpl \
           -t 'JUNO visible spectra with $\\sigma_E=3\\%$ @ 1 MeV' \
           --xlabel '$E_\\mathrm{vis}$, MeV' \
           --ylabel 'Entries/MeV' \
           --xlim 1.5 5.0 \
           --ylim 1.5e4 \
           --grid \
    -- mpl \
           -s
"""

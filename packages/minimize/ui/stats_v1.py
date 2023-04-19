"""Build test statistic based on arbitrary sum of χ² and logPoisson functions

Based on stats UI. Notes:
    - works with minimizer-v2 at least
"""

from gna.ui import basecmd
from gna import constructors as C

class cmd(basecmd):
    components: list
    storage: list
    identities: list
    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('name', help='statistic name')
        parser.add_argument('-c', '--chi2',        default=[], action='append', type=env.parts.analysis, help='χ² contribution')
        parser.add_argument('--chi2-cnp-stat',     default=[], action='append', type=env.parts.analysis, help='χ² CNP (stat) contribution')
        parser.add_argument('--chi2-unbiased',     default=[], action='append', type=env.parts.analysis, help='χ² contribution, bias corrected (+ln|V|)')
        parser.add_argument('-p', '--logpoisson',  default=[], action='append', type=env.parts.analysis, help='logPoisson contribution')
        parser.add_argument('-P', '--logpoisson-ratio',  default=[], action='append', type=env.parts.analysis, help='log(Poisson ratio) contribution')
        parser.add_argument('--chi2-nuisance',     default=[], action='append', type=env.parts.analysis, help='χ² contribution (nuisance)')
        parser.add_argument('--labels', nargs='*', default=[], action='append', help='Node labels')
        parser.add_argument('--debug-min-steps', action='store_true', help='Debug minimization steps')

        parser.add_argument('--legacy', action='store_true', help='use legacy storage')

    def init(self):
        self.components = []
        self.storage = []
        self.identities = []

        self._labels = self.opts.labels.copy()
        self.ndf_data = 0
        self.ndf_nuisance = 0

        self._map_analysis(self._add_chi2, self.opts.chi2)
        self._map_analysis(self._add_chi2_unbiased, self.opts.chi2_unbiased)
        self._map_analysis(self._add_chi2_cnp_stat, self.opts.chi2_cnp_stat)
        self._map_analysis(self._add_logpoisson, self.opts.logpoisson)
        self._map_analysis(self._add_logpoisson_ratio, self.opts.logpoisson_ratio)
        self._map_analysis(self._add_chi2, self.opts.chi2_nuisance, add_to_ndf=False)

        if len(self.components)==1:
            self.statistic = self.components[0]
        else:
            clabel = self._labels and self._labels[0] or 'Statistic'
            sm = C.Sum(self.components, labels=clabel)
            self.statistic = self.debug_append('statistic', sm.sum.sum)

        if self.opts.legacy:
            self.env.parts.statistic[self.opts.name] = self.statistic

        self.env.future[('statistic', self.opts.name)] = {
                'fun': self.statistic,
                'ndf_data': self.ndf_data,
                'ndf_nuisance': self.ndf_nuisance,
                }

    def _mklabel(self, label: str, i: int, n: int) -> str:
        if n>1:
            if label:
                return f'{label} ({i}/{n})'
            else:
                return f'({i}/{n})'

        return label

    def _map_analysis(self, fcn_add, analyses_list, add_to_ndf: bool=True):
        if self._labels:
            label, self._labels = self._labels[0], self._labels[1:]
        else:
            label = ''

        if not add_to_ndf:
            label = f'nuisance {label}'

        n_ana = len(analyses_list)

        for self.i_ana, analysis in enumerate(analyses_list):
            label_ana = self._mklabel(label, self.i_ana, n_ana)
            n_blocks = len(analysis)

            for self.i_block, block in enumerate(analysis):
                label_block = self._mklabel('', self.i_block, n_blocks)
                fcn_add(block.theory, block.data, block.cov, label=f'{label_ana}{label_block}'.strip())

                if add_to_ndf:
                    self.ndf_data += block.data.data().shape[0]
                else:
                    self.ndf_nuisance += block.data.data().shape[0]

    def _add_chi2(self, theory, data, cov, label: str):
        clabel = f'χ² {label}'.strip()
        chi2 = C.Chi2(labels=clabel)
        chi2.add(theory, data, cov)

        self.storage.append(chi2)
        self.debug_append('chi2', chi2.chi2.chi2)

    def _add_chi2_unbiased(self, theory, data, cov, label: str):
        clabel1 = f'χ² {label}'.strip()
        clabel2 = f'ln\\|V\\| {label}'.strip()
        clabel3 = f'χ²+ln\\|V\\| {label}'.strip()

        chi2 = C.Chi2(labels=clabel1)
        logproddiag = C.LogProdDiag(labels=clabel2)

        chi2s = self.debug('chi2_chi2', chi2.single())
        logproddiags = self.debug('chi2_lnV', logproddiag.single())
        chi2u = C.Sum([chi2s, logproddiags], labels=clabel3)

        chi2.add(theory, data, cov)
        logproddiag.add(cov)

        self.storage.extend((chi2, logproddiag, chi2u))
        self.debug_append('chi2_unbias', chi2u.single())

    def _add_chi2_cnp_stat(self, theory, data, cov, label: str):
        clabel = f'χ²(CNP) stat {label}'.strip()
        chi2 = C.Chi2CNPStat(labels=clabel)
        chi2.add(theory, data)

        self.storage.append(chi2)
        self.debug_append('chi2_cnp_stat', chi2.chi2.chi2)

    def _add_logpoisson(self, theory, data, cov, label: str):
        clabel = f'log(Poisson) {label}'.strip()
        logpoisson = C.LogPoissonSplit(False, labels=(clabel+' (const)', clabel))
        logpoisson.add(theory, data)

        self.storage.append(logpoisson)
        self.debug_append('logpoisson', logpoisson.poisson.poisson)

    def _add_logpoisson_ratio(self, theory, data, cov, label: str):
        clabel = f'log(Poisson ratio) {label}'.strip()
        logpoisson_ratio = C.LogPoissonSplit(True, labels=(clabel+' (const)', clabel))
        logpoisson_ratio.add(theory, data)

        self.storage.append(logpoisson_ratio)
        self.debug_append('logpoisson_r', logpoisson_ratio.poisson.poisson)

    def debug_append(self, label: str, output):
        newoutput = self.debug(label, output)
        self.components.append(newoutput)

        return newoutput

    def debug(self, label: str, output):
        if not self.opts.debug_min_steps:
            return output

        ident = C.IdentityVerbose(f'{label}_{self.i_ana}_{self.i_block}')
        self.identities.append(ident)
        return ident.add_input(output)


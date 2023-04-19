"""Build test statistic based on arbitrary sum of χ² and logPoisson functions"""

from gna.ui import basecmd
import ROOT
from gna import constructors as C

class cmd(basecmd):
    components: list
    storage: list
    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('name', help='statistic name')
        parser.add_argument('-c', '--chi2',        default=[], action='append', type=env.parts.analysis, help='χ² contribution')
        parser.add_argument('--chi2-cnp-stat',     default=[], action='append', type=env.parts.analysis, help='χ² CNP (stat) contribution')
        parser.add_argument('--chi2-unbiased',     default=[], action='append', type=env.parts.analysis, help='χ² contribution, bias corrected (+ln|V|)')
        parser.add_argument('-p', '--logpoisson',  default=[], action='append', type=env.parts.analysis, help='logPoisson contribution')
        parser.add_argument('-P', '--logpoisson-ratio',  default=[], action='append', type=env.parts.analysis, help='log(Poisson ratio) contribution')
        parser.add_argument('--logpoisson-legacy', default=[], action='append', type=env.parts.analysis, help='logPoisson contribution (legacy)')
        parser.add_argument('--labels', nargs='+', default=[], help='Node labels')

    def init(self):
        self.components = []
        self.storage = []
        labels = list(reversed(self.opts.labels))

        self.ndf_chi2 = 0

        if self.opts.chi2:
            clabel = labels and labels.pop() or 'χ²'
            chi2 = C.Chi2(labels=clabel)
            for analysis in self.opts.chi2:
                for block in analysis:
                    chi2.add(block.theory, block.data, block.cov)
                    self.ndf_chi2 += block.data.data().shape[0]

            self.storage.append(chi2)
            self.components.append(chi2)

        if self.opts.chi2_unbiased:
            clabel = labels and labels.pop() or 'χ²'
            llabel = labels and labels.pop() or 'logdetV'
            chi2 = C.Chi2(labels=clabel)
            logproddiag = C.LogProdDiag(labels=llabel)
            chi2u = C.Sum([chi2.single(), logproddiag.single()], labels=f'{clabel}+{llabel}')
            for analysis in self.opts.chi2_unbiased:
                for block in analysis:
                    chi2.add(block.theory, block.data, block.cov)
                    logproddiag.add(block.cov)
                    self.ndf_chi2 += block.data.data().shape[0]

            self.storage.append(chi2)
            self.storage.append(logproddiag)
            self.storage.append(chi2u)
            self.components.append(chi2u)

        if self.opts.chi2_cnp_stat:
            clabel = labels and labels.pop() or 'χ²(CNP)'
            chi2 = C.Chi2CNPStat(labels=clabel)
            for analysis in self.opts.chi2_cnp_stat:
                for block in analysis:
                    chi2.add(block.theory, block.data)
                    self.ndf_chi2 += block.data.data().shape[0]

            self.storage.append(chi2)
            self.components.append(chi2)

        if self.opts.logpoisson_legacy:
            clabel = labels and labels.pop() or 'logPoisson'
            logpoisson_legacy = C.Poisson(False, labels='')
            for analysis in self.opts.logpoisson_legacy:
                for block in analysis:
                    logpoisson_legacy.add(block.theory, block.data)

            self.components.append(logpoisson_legacy)

        if self.opts.logpoisson:
            clabel = labels and labels.pop() or 'logPoisson'
            logpoisson = C.LogPoissonSplit(False, labels=(clabel+' (const)', clabel))
            for analysis in self.opts.logpoisson:
                for block in analysis:
                    logpoisson.add(block.theory, block.data)

            self.components.append(logpoisson)

        if self.opts.logpoisson_ratio:
            clabel = labels and labels.pop() or 'log(Poisson ratio)'
            logpoisson_ratio = C.LogPoissonSplit(True, labels=(clabel+' (const)', clabel))
            for analysis in self.opts.logpoisson_ratio:
                for block in analysis:
                    logpoisson_ratio.add(block.theory, block.data)

            self.components.append(logpoisson_ratio)

        if len(self.components)==1:
            self.statistic = self.components[0]
        else:
            clabel = labels and labels.pop() or ''
            self.statistic = C.Sum([comp.transformations.back() for comp in self.components], labels=clabel)

        self.env.parts.statistic[self.opts.name] = self.statistic
        if self.ndf_chi2 != 0:
            self.env.future.child(("ndf"))[self.opts.name] = self.ndf_chi2

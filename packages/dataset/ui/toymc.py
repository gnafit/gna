"""Make a ToyMC based on an output"""

from gna.ui import basecmd
from gna import constructors as C
import ROOT as R

class cmd(basecmd):
    _cholesky = None

    @classmethod
    def initparser(cls, parser, _):
        parser.add_argument('-r', '--root', default='spectra', help='root location for read/store actions')
        parser.add_argument('name_in', help='input observable name')
        parser.add_argument('name_out', help='observable name (output)')
        parser.add_argument('-l', '--label', help='ToyMC node label')
        parser.add_argument('-t', '--type', required=True, choices=('asimov', 'poisson', 'normal', 'normalStats', 'covariance'), help='MC type')

        unc = parser.add_mutually_exclusive_group()
        unc.add_argument('-u', '--uncertainties', help='1d array with uncertinainties or Cholesky decomposed cov matrix (normal/covariance)' )
        unc.add_argument('-c', '--covariance', help='1d diagonal or 2d full covariance matrix')

    def __init__(self, *args, **kwargs):
        basecmd.__init__(self, *args, **kwargs)

    def init(self):
        output = self.env.future[self.opts.root, self.opts.name_in]

        if not output:
            raise Exception('Invalid or missing output: {}'.format(self.opts.name_in))

        if self.opts.type == 'poisson':
            self.toymc = C.PoissonToyMC()
            self.toymc.add(output)
        elif self.opts.type == 'normalStats':
            self.toymc = C.NormalStatsToyMC()
            self.toymc.add_input(output)
        elif self.opts.type == 'asimov':
            self.toymc = C.Snapshot()
            self.toymc.add_input(output)
        else:
            unc_l = self._get_uncertainties()
            if self.opts.type == 'covariance':
                self.toymc = C.CovarianceToyMC(True, R.GNA.MatrixFormat.PermitDiagonal)
                self.toymc.add(output, unc_l)
            elif self.opts.type == 'normal':
                self.toymc = C.NormalToyMC()
                self.toymc.add(output, unc_l)
            else:
                raise Exception(f"Invalid ToyMC type: {self.opts.type}")

        trans = self.toymc.transformations[0]
        trans.setLabel(f'ToyMC: {self.opts.type}' if self.opts.label is None else self.opts.label)
        trans.touch()
        self.env.future[self.opts.root, self.opts.name_out] = trans.single()

    def _get_uncertainties(self):
        if self.opts.uncertainties:
            return self.env.future[self.opts.root, self.opts.uncertainties]

        if not self.opts.covariance:
            raise Exception(f'The uncertainties/covariance for {self.opts.type} ToyMC are not specified')

        cov = self.env.future[self.opts.root, self.opts.covariance]
        self._cholesky = C.Cholesky(R.GNA.MatrixFormat.PermitDiagonal, labels='Cholesky')
        cov >> self._cholesky.cholesky.mat
        return self._cholesky.cholesky.L


"""Save values of a group of parameters"""

from gna.ui import basecmd
from env.lib.cwd import update_namespace_cwd
from gna import constructors as C
import ROOT as R
import numpy as np

class cmd(basecmd):
    _printoptions = None
    @classmethod
    def initparser(cls, parser, _):
        parser.add_argument('groups', nargs='+', help='parameter groups to work with')
        parser.add_argument('-v', '--verbose', action='count', default=0, help='increase verbosity level')
        parser.add_argument('--edgeitems', type=int, help='number of rows/columns to print at the begin/end of arrays')
        parser.add_argument('-m', '--mode', default='normal', choices=('covariance', 'asimov'), help='parameter MC mode')

    def init(self):
        if self.opts.verbose:
            print(f'Pargroup MC {self.opts.mode}: {self.opts.groups}')
        if self.opts.mode=='asimov':
            return

        self.objects={}
        self.namespaces=[]
        groups_loc = self.env.future['parameter_groups']

        self._print_pre()

        self.mc = C.CovarianceToyMC(True, R.GNA.MatrixFormat.PermitDiagonal, labels=f'Pars MC')
        self.parinput = C.ParArrayInput(labels=f'Pars input')

        output = dict()
        for name in self.opts.groups:
            group = groups_loc[name]
            self._process_group(name, group)

        self.parinput.materialize()

        # self.mc.printtransformations()
        # self.parinput.printtransformations()

        self.parinput.pararray.touch()

        self._print_post()

    def _process_group(self, name, group):
        self.parinput.add_input()

        centers = C.ParCenters(labels=f'Pars centers: {name}')
        covmat = C.ParCovMatrix(R.GNA.MatrixFormat.PermitDiagonal, labels=f'Pars covmat: {name}')
        cholesky = C.Cholesky(R.GNA.MatrixFormat.PermitDiagonal, labels=f'Cholesky: {name}')

        self.objects[name] = {'centers': centers, 'covmat': covmat, 'cholesky': cholesky}

        for par in group.values():
            centers.append(par)
            covmat.append(par)
            self.parinput.append(par)
        centers.materialize()
        covmat.materialize()

        covmat.unc_matrix >> cholesky.cholesky.mat
        self.mc.add(centers.centers, cholesky)
        self.mc.toymc.outputs.back() >> self.parinput.pararray.inputs.back()

        if self.opts.verbose>1:
            print(f'Group {name} centers, covmat, Cholesky:')
            print(centers.centers.data())
            print(covmat.unc_matrix.data())
            print(cholesky.cholesky.data())

    def _print_pre(self):
        if self.opts.verbose<2 or self.opts.edgeitems is None:
            return

        self._printoptions = np.get_printoptions()
        np.set_printoptions(edgeitems=self.opts.edgeitems)

    def _print_post(self):
        if self._printoptions is None:
            return

        np.set_printoptions(**self._printoptions)

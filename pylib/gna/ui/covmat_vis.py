from __future__ import print_function
from gna.env import env
from gna.ui import basecmd
import ROOT
import numpy as np
import matplotlib.pyplot as plt
from mpl_tools.helpers import savefig

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('--analysis', type=env.parts.analysis, required=True,
                            help='Analysis from which covmatrices would be used')
        parser.add_argument('--savefig', '-o', '--output', help='Path to save figure')
        parser.add_argument('--show', action='store_true',
                                  help='Show plot of covariance matrix')
        parser.add_argument('--dump', help='File to dump covariance matrix')
        parser.add_argument('--mask', action='store_true',
                             help="Mask zeros from covariance matrix")


    def init(self):
        chol_blocks = (np.tril(block.cov.data()) for block in self.opts.analysis)
        matrix_stack = [np.matmul(chol, chol.T) for chol in chol_blocks]
        covmat = self.make_blocked_matrix(matrix_stack)
        sdiag = np.diagonal(covmat)**0.5
        cormat = covmat/sdiag/sdiag[:,None]

        if self.opts.mask:
            covmat = np.ma.array(covmat, mask=(covmat == 0.))
            cormat = np.ma.array(cormat, mask=(cormat == 0.))

        fig, ax = plt.subplots()
        im = ax.matshow(covmat)
        ax.minorticks_on()
        cbar = fig.colorbar(im)
        plt.title("Covariance matrix")

	savefig(self.opts.savefig, suffix='_cov')

        fig, ax = plt.subplots()
        im = ax.matshow(cormat)
        ax.minorticks_on()
        cbar = fig.colorbar(im)
        plt.title("Correlation matrix")

	savefig(self.opts.savefig, suffix='_cor')

        if self.opts.dump:
            np.savez(self.opts.dump, covmat)

        if self.opts.show:
            plt.show()

    def make_blocked_matrix(self, matrices):
        matrix_stack = []
        total_size = sum(mat.shape[0] for mat in matrices)
        for idx, matrix in enumerate(matrices):
            size_to_left = sum(mat.shape[1] for mat in matrices[:idx])
            assert size_to_left is not None
            size_to_right = total_size - size_to_left - matrix.shape[1]
            layer = [np.zeros((matrix.shape[0], size_to_left)), matrix, np.zeros((matrix.shape[0], size_to_right))]
            matrix_stack.append(layer)
        return np.block(matrix_stack)


import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
import ROOT
from gna.env import env
from gna.ui import basecmd
from mpl_tools.helpers import savefig

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        mode = parser.add_mutually_exclusive_group(required=True)
        mode.add_argument('--fit-input', type=os.path.abspath,
             help='Path to file with covariance matrix produced by minimizer after the fit')
        mode.add_argument('--analysis', type=env.parts.analysis,
                            help='Analysis from which covmatrices would be used')

        parser.add_argument('--fit-parameters', action='append', nargs='*', default=[],
             help='Keep only covariances for specified parameters from file')
        parser.add_argument('--savefig', '-o', '--output', help='Path to save figure')
        parser.add_argument('--cmap', help='Use cmap from matplotlib')
        parser.add_argument('--show', action='store_true',
                             help='Show plot of covariance matrix')
        parser.add_argument('--dump', help='File to dump covariance matrix')
        parser.add_argument('--mask', action='store_true',
                             help="Mask zeros from covariance matrix")

    def init(self):
        if self.opts.fit_input:
            self.from_fit()
        if self.opts.analysis:
            self.from_graph()

        if self.opts.mask:
            self.mask_zeroes()

        self.plot_matrices()


    def from_fit(self):
        with h5py.File(self.opts.fit_input, 'r') as f:
            parameters_from_file = f['par_names']
            self.covmat = f['cov_matrix'][:]
            sigmas = np.diagonal(self.covmat)**0.5
            self.cormat = self.covmat/sigmas/sigmas[:,None]


    def from_graph(self):
        chol_blocks = (np.tril(block.cov.data()) for block in self.opts.analysis)
        matrix_stack = [np.matmul(chol, chol.T) for chol in chol_blocks]
        self.covmat = self.make_blocked_matrix(matrix_stack)
        sdiag = np.diagonal(covmat)**0.5
        self.cormat = covmat/sdiag/sdiag[:,None]

    def plot_matrices(self):
        if self.opts.cmap:
            plt.set_cmap(self.opts.cmap)

        fig, ax = plt.subplots()
        im = ax.matshow(self.covmat)
        ax.minorticks_on()
        cbar = fig.colorbar(im)
        plt.title("Covariance matrix")

        savefig(self.opts.savefig, suffix='_cov')
        fig, ax = plt.subplots()
        im = ax.matshow(self.cormat)
        ax.minorticks_on()
        cbar = fig.colorbar(im)
        plt.title("Correlation matrix")

        savefig(self.opts.savefig, suffix='_cor')

        if self.opts.dump:
            np.savez(self.opts.dump, self.covmat)

        if self.opts.show:
            plt.show()

    def mask_zeroes(self):
        self.covmat = np.ma.array(self.covmat, mask=(self.covmat == 0.))
        self.cormat = np.ma.array(self.cormat, mask=(self.cormat == 0.))

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

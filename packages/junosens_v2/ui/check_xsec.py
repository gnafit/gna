""" Check the GNA crossection via external one, provided with ROOT file
"""
from gna.ui import basecmd
from tools.classwrapper import ClassWrapper
import gna.env
from tabulate import tabulate
from env.lib.cwd import update_namespace_cwd
from tools.root_helpers import TFileContext
from mpl_tools.helpers import pcolormesh, savefig
import numpy as np
from matplotlib import pyplot as plt
from junosens_v2.lib.makefcn import MakeFcn2

class check_xsec(basecmd):
    _instance_count = [-1]
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._instance_count[0]+=1
        self._instance=self._instance_count[0]

    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('exp', help='JUNO exp instance')
        parser.add_argument('-i', '--input', type=TFileContext.type(), required=True, help='input ROOT file')
        parser.add_argument('-N', '--name-suffix', default='', help='object name suffix')
        parser.add_argument('-o', '--output', nargs='+', default=[], help='output file')
        parser.add_argument('-a', '--group-a', default='GNA', help='group A')
        parser.add_argument('-b', '--group-b', default='other', help='group B')

    def init(self):
        update_namespace_cwd(self.opts, 'output')
        try:
            self.exp = self.env.parts.exp[self.opts.exp]
        except Exception:
            print('Unable to retrieve exp '+self.opts.exp)

        self.namespace = self.exp.namespace
        self.context   = self.exp.context
        self.outputs   = self.exp.context.outputs

        # evis        = self.outputs.evis.data()
        # ctheta      = self.outputs.ctheta.data()
        # evis_mesh   = self.outputs.evis_mesh.data()
        # ctheta_mesh = self.outputs.ctheta_mesh.data()
        # ee          = self.outputs.ee.data()
        # enu         = self.outputs.enu.data()
        # xsec        = self.outputs.ibd_xsec.data()
        # jac         = self.outputs.jacobian.data()

        with self.opts.input as f:
            dsigma_dcos_Enu_cos=f.Get('dsigma_dcos_Enu_cos'+self.opts.name_suffix)
            epos_vs_enu_cos=f.Get('Epositron_Enu_cos'+self.opts.name_suffix)
            if not dsigma_dcos_Enu_cos or not epos_vs_enu_cos:
                f.ls()
                assert False

        #
        # Define mesh for testing
        #
        enu    = np.linspace( 1.0, 10.0, 51)
        ctheta = np.linspace(-1.0, 1.0,  51)
        Enu, Ctheta = np.meshgrid(enu, ctheta, indexing='ij')

        enuc = (enu[1:]+enu[:-1])*0.5
        cthetac = (ctheta[1:]+ctheta[:-1])*0.5
        Enuc, Cthetac = np.meshgrid(enuc, cthetac, indexing='ij')

        #
        # Define vectorized function
        #
        XsecFcnIn = np.vectorize(dsigma_dcos_Enu_cos.Eval, otypes='d')
        EposFcnIn = np.vectorize(epos_vs_enu_cos.Eval, otypes='d')

        XsecFcn = MakeFcn2(self.outputs.enu, self.outputs.ctheta_mesh, self.outputs.ibd_xsec)

        #
        # Calculate on mesh
        #
        xsec_in = XsecFcnIn(Enuc, Cthetac)
        epos_in = EposFcnIn(Enuc, Cthetac)

        xsec = XsecFcn(Enuc, Cthetac)

        #
        # Apply mask for the ratio
        #
        mask = (xsec_in==0)*(xsec==0)
        ratio = xsec/xsec_in
        ratio[mask]=1.0
        lratio = np.log(ratio)
        lratio[lratio<-10]=-10
        diff = xsec-xsec_in

        #
        # Cross sections
        #
        vmin = np.minimum(xsec, xsec_in).min()
        vmax = np.maximum(xsec, xsec_in).max()
        meshopts = dict(shading='auto', vmin=vmin, vmax=vmax, rasterized=bool(self.opts.output))

        fig = plt.figure(f"{self._instance}-xsec-gna")
        ax = plt.subplot(111, xlabel=r'$E_{\nu}$, MeV', ylabel=r'$\cos\theta$', title=f'Cross section: {self.opts.group_a}')
        c=pcolormesh(Enu, Ctheta, xsec, colorbar=True, **meshopts)
        savefig(self.opts.output, suffix='_xsec_a')

        fig = plt.figure(f"{self._instance}-xsec-in")
        ax = plt.subplot(111, xlabel=r'$E_{\nu}$, MeV', ylabel=r'$\cos\theta$', title=f'Cross section: {self.opts.group_b}')
        c=pcolormesh(Enu, Ctheta, xsec_in, colorbar=True, **meshopts)
        savefig(self.opts.output, suffix='_xsec_b')

        #
        # Diffs
        #
        meshopts = dict(shading='auto', rasterized=bool(self.opts.output))
        fig = plt.figure(f"{self._instance}-xsec-ratio-free")
        ax = plt.subplot(111, xlabel=r'$E_{\nu}$, MeV', ylabel=r'$\cos\theta$', title=f'Cross section comparison: log({self.opts.group_a}/{self.opts.group_b})')
        c=pcolormesh(Enu, Ctheta, lratio, colorbar=True, **meshopts)
        savefig(self.opts.output, suffix='_xsec_lratio_free')

        fig = plt.figure(f"{self._instance}-xsec-ratio")
        ax = plt.subplot(111, xlabel=r'$E_{\nu}$, MeV', ylabel=r'$\cos\theta$', title=f'Cross section comparison: log({self.opts.group_a}/{self.opts.group_b})')
        c=pcolormesh(Enu, Ctheta, lratio, colorbar=True, vmin=-0.03, vmax=0.01, **meshopts)
        savefig(self.opts.output, suffix='_xsec_lratio')

        fig = plt.figure(f"{self._instance}-xsec-ratio-distr-free")
        ax = plt.subplot(111, xlabel=f'log({self.opts.group_a}/{self.opts.group_b})', ylabel=r'entries', title='Cross section comparison')
        ax.hist(lratio.ravel(), bins=100)
        savefig(self.opts.output, suffix='_xsec_lratio_distr_free')

        fig = plt.figure(f"{self._instance}-xsec-ratio-distr")
        ax = plt.subplot(111, xlabel=f'log({self.opts.group_a}/{self.opts.group_b})', ylabel=r'entries', title='Cross section comparison')
        vmin, vmax = -0.03, 0.01
        under, over = (lratio<vmin).sum(), (lratio>=vmax).sum()
        ax.hist(lratio.ravel(), bins=100, range=(vmin, vmax), label=f'under/over: {under}/{over}')
        ax.legend(loc='upper right')
        savefig(self.opts.output, suffix='_xsec_lratio_distr')

        # fig = plt.figure(f"{self._instance}-xsec-ratio-e")
        # ax = plt.subplot(111, xlabel=r'$E_{\nu}$, MeV', ylabel=f'log(GNA/{self.opts.group})', title='Cross section comparison')
        # for i, c in enumerate(ctheta):
            # ax.plot(evis, lratio[:,i], label=f'{c}')
        # ax.legend(loc='upper right')
        # ax.set_ylim(-0.04, 0.04)
        # savefig(self.opts.output, suffix='_xsec_lratio_vs_e')

        # fig = plt.figure(f"{self._instance}-xsec-ratio-ct")
        # ax = plt.subplot(111, xlabel=r'$\cos\theta$', ylabel=f'log(GNA/{self.opts.group})', title='Cross section comparison')
        # stride=800
        # for i, e in enumerate(evis[::stride]):
            # ax.plot(ctheta, lratio[stride,:], 'o-', label=f'{e:02f}')
        # ax.legend(loc='upper left', title='Evis, MeV')
        # # ax.set_ylim(-0.04, 0.04)
        # savefig(self.opts.output, suffix='_xsec_lratio_vs_c')

        fig = plt.figure(f"{self._instance}-xsec-diff")
        ax = plt.subplot(111, xlabel=r'$E_{\nu}$, MeV', ylabel=r'$\cos\theta$', title=f'Cross section comparison: {self.opts.group_a}-{self.opts.group_b}')
        c=pcolormesh(Enu, Ctheta, diff, colorbar=True, **meshopts)
        savefig(self.opts.output, suffix='_xsec_diff')

        # #
        # # Energy
        # #
        # fig = plt.figure(f"{self._instance}-xsec-ratio-e")
        # ax = plt.subplot(111, xlabel='GNA, MeV', ylabel=rf'{self.opts.group}/GNA', title='Positron energy')
        # for i, c in enumerate(ctheta):
            # ax.plot(ee, epos_in[:,i]/ee, label=f'{c}')
        # ax.legend(title=r'$\cos\theta$')
        # ax.set_ylim(0.99999, 1.00001)
        # savefig(self.opts.output, suffix='_epos_ratio')

        # fig = plt.figure(f"{self._instance}-xsec-e")
        # ax = plt.subplot(111, xlabel='GNA, MeV', ylabel=rf'{self.opts.group}, MeV', title='Positron energy')
        # for i, c in enumerate(ctheta):
            # ax.plot(ee, epos_in[:,i], label=f'{c}')
        # ax.legend(title=r'$\cos\theta$')
        # ax.set_xlim(0.510, 0.517)
        # ax.set_ylim(0.0, 0.55)
        # savefig(self.opts.output, suffix='_epos')

        # fig = plt.figure(f"{self._instance}-xsec-epos")
        # ax = plt.subplot(111, xlabel='Enu, MeV', ylabel=r'Epos, MeV', title='Positron energy')
        # for i, c in enumerate(ctheta):
            # ax.plot(enu[:, i], ee, label=f'{c}')
            # ax.plot(enu[:, i], epos_in[:,i], label=f'{c} (in)')
        # ax.legend(title=r'$\cos\theta$')
        # # ax.set_ylim(0.99999, 1.00001)


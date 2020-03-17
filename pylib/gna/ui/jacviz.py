from gna.ui import basecmd
from gna import constructors as C
import os
import h5py
import numpy as np
import matplotlib.ticker as mtick
from matplotlib import pyplot as plt
from mpl_tools.plot_lines import plot_lines
from gna.parameters.parameter_loader import get_parameters

class cmd(basecmd):
    """
Jacviz -- module to build and visualize jacobian matrix, 
systematic/statistical errors covariation and correlation matrices 
for chosen groups of parameters of the model

Simple example of using:
    ./gna -- exp --ns H0 dayabay_p15a \         -- used model of experiment
	  -- jacviz --ns H0 -params \           -- calling jacviz module
	  --groups bkg_rate_fastn bkg_rate_amc bkg_rate_alphan bkg_rate_lihe \  -- the first group of parameters
	  --groups eres lsnl_weight escale frac_li OffdiagScale \               -- the second group of parameters
	  --groups fission_fraction_corr eper_fission \                         
          nominal_thermal_power fastn_shape acc_norm effunc_uncorr pmns \       -- the third group of parameters
	  --gnames bkg_unc energy_scale reactor_unc \                           -- custom names of chosen groups
	  --out-fig tmp/dyb/jacviz/dyb.pdf --out-hdf5 tmp/dyb/jacviz/dyb.hdf5   -- save output data and plots
    """
    def DataSaver(self, gnames):
        data = {}
        data['prediction'] = None
        for gname in gnames:
            data[gname] = {'jac': None}
            data[gname] = {'syst': None}
            data[gname] = {'diag': None}
            data[gname] = {'full': None}
            data[gname] = {'syst_corl': None}
            data[gname] = {'full_corl': None}
            data[gname] = {'Chol': None}
            data[gname] = {'params': None}
        self.data = data
        self.data_names = {'jac': 'Jacobian', 'syst': 'Systematic covariance matrix',
                'full': 'Full covariance matrix', 'diag': 'diagonal',
                'syst_corl': 'Correlation matrix: systematic',
                'full_corl': 'Correlation matrix: full', 'Chol': 'Cholesky decomposition'}
 
    def make_jac(self, prediction, cov_pars, gname):
        jac = C.Jacobian()
        par_covs = C.ParCovMatrix()
        for par in cov_pars:
            jac.append(par)
            par_covs.append(par)
        prediction >> jac.jacobian.func
        par_covs.materialize()
        jac.jacobian.setLabel('jac_'+gname)
        return jac.jacobian, par_covs.unc_matrix

    def plot_matrix(self, fig, ax, data, gname, key):
        data = np.ma.array(data, mask=(data==0.0))
        im = ax.matshow(data, aspect=float(data.shape[1])/data.shape[0])
        ax.ticklabel_format(axis='both', useMathText=True)
        ax.set_xlim(-.5, data.shape[1]-.5)
        ax.set_ylim(-.5, data.shape[0]-.5)
        cb = fig.colorbar(im)
        cb.formatter.set_powerlimits((0, 0))
        cb.ax.yaxis.set_offset_position('right')
        cb.update_ticks()
        fig.gca().xaxis.tick_bottom()
        plt.title(self.data_names[key]+' for '+gname, fontsize=18)

    def plot_diag(self, fig, ax, data, gname, key):
        x_axis = [i for i in range(data.shape[0])]
        im = ax.step(x_axis, data**.5, where='mid')
        ax.set_ylabel('abs stat error', fontsize=16)
        ax.set_xlim(x_axis[0], x_axis[-1])
        ax.set_ylim(0., ax.get_ylim()[1])
        ax2 = ax.twinx()
        ax2.set_ylabel('rel stat error', color='red', fontsize=16)
        ax2.step(x_axis, data**.5/data*100, where='mid', color='red')
        ax2.set_xlim(x_axis[0], x_axis[-1])
        ax2.set_ylim(0., ax2.get_ylim()[1])
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
        plt.title('Absolute and relative {} errors{}'.format(key, gname), fontsize=18)


    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('-n', '--ns', help='Namespece of inited model')
        parser.add_argument('-p', '--params', action='store_true', help='Print parameters in table')
        parser.add_argument('-g', '--groups', nargs='*', action='append', 
                help='Namespace of parameters for jacobian')
        parser.add_argument('--gnames', nargs='*', help='Names of groups of parameters')
        parser.add_argument('--out-hdf5', help='path/to/output.hdf5')
        parser.add_argument('--out-fig', help='path/to/output/figures')
        parser.add_argument('--observables', nargs='*', help='Observables for stat. errors')
        parser.add_argument('-s', '--show', action='store_true', help='Show figures')

    def run(self):
        groups = self.opts.groups
        if self.opts.gnames:
            gnames = self.opts.gnames
            if len(gnames) != len(groups):
                raise "Length of groups and name of groups don\'t equal"
        else:
            gnames = [group[0] for group in groups]
        self.DataSaver(gnames)
        if self.opts.ns:
            ns = self.env.ns(self.opts.ns)
        else:
            ns = self.env.globalns
        observables = []
        prediction = C.Concat()
        grid = []
        if self.opts.observables:
            ns = self.env.globalns
            for obs in self.opts.observables:
                observables.append((obs, ns.getobservable(obs)))
                prediction.append(ns.getobservable(obs))
                grid.append(ns.getobservable(obs).data().shape[0])
                print(obs+' added')
            splited = obs.split('/')
        else:
            for obs in ns.walkobservables():
                splited = obs[0].split('/')
                if len(splited) == 2:
                    short = splited[1]
                    if '.' not in short and '_' not in short:
                        observables.append(obs)
                        prediction.append(obs[1])
                        grid.append(obs[1].data().shape[0])
                        print(obs[0]+' added')
        self.data['prediction'] = prediction.data().copy()
        name = splited[0]
	covmat = C.Covmat()
	covmat.cov.stat.connect(prediction)
        covmat.cov.setLabel('Covmat')
        for group, gname in zip(groups, gnames):
            cov_pars = get_parameters([name+'.'+g for g in group], drop_fixed=True, drop_free=True)
            cov_names = [x.qualifiedName()[len(name)+1:] for x  in cov_pars]
            jac, par_covs = self.make_jac(prediction, cov_pars, gname)

            product = np.matmul(jac.data().copy(), par_covs.data().copy())
            product = np.matmul(product.copy(), jac.data().copy().T)
            jac_norm = jac.data().T / prediction.data()
            self.data[gname]['jac'] = jac_norm.T
            self.data[gname]['syst'] = syst = product.copy()
            self.data[gname]['full'] = cov_full = covmat.cov.data().copy() + syst
            self.data[gname]['diag'] = np.diagonal(cov_full.copy())
            self.data[gname]['Chol'] = np.linalg.cholesky(cov_full)
            sdiag = np.diagonal(syst)**.5
            self.data[gname]['syst_corl'] = syst / sdiag / sdiag[:, None]
            sdiag = np.diagonal(cov_full)**.5
            self.data[gname]['full_corl'] = cov_full / sdiag / sdiag[:, None]
            self.data[gname]['params'] = group
            for gname in gnames:
                if any(self.data[gname]) is None:
                    raise "None {} for {}".format(self.data_names[key], gname)

        if self.opts.out_hdf5:
            path = self.opts.out_hdf5
            with h5py.File(path, 'w') as f:
                f.create_dataset('prediction', data=self.data['prediction'])
                for gname in self.data.keys():
                    if gname != 'prediction':
                        for key in self.data[gname].keys():
                            f.create_dataset(gname+'/'+key, data=self.data[gname][key])
            print('Save output file: '+path)


        if any([self.opts.out_fig, self.opts.show]):
            num = 0
            path = str(self.opts.out_fig).split('.')
            fig, ax = plt.subplots(figsize=(12, 9), dpi=300)
            data = self.data['prediction']**0.5
            self.plot_diag(fig, ax, data, '', 'stat.')
            fig.tight_layout()
            if self.opts.out_fig:
                tmp_path = path[0]+'_{:02}_stat.{}'.format(num, path[1])
                plt.savefig(tmp_path)
                num += 1
                print('Save output file: '+tmp_path)
            if self.opts.show:
                plt.show()
            fig.clf()
            fig.clear()
            plt.close()
            for gname in gnames:
                for key in self.data_names.keys():
                    if key == 'jac' and self.opts.params:
                        fig, ax = plt.subplots(figsize=(15, 9), dpi=300)
                    else:
                        fig, ax = plt.subplots(figsize=(12, 9), dpi=300)
                    data = self.data[gname][key]
                    if key == 'diag':
                        self.plot_diag(fig, ax, data, ' for '+gname, key)
                    else:
                        self.plot_matrix(fig, ax, data, gname, key)
                    if key not in {'jac', 'diag'}:
                        for i in range(1, len(grid)):
                            lvl = i * grid[i] -.5
                            ax.plot([0, data.shape[0]], [lvl, lvl], color='white', alpha=0.5)
                            ax.plot([lvl, lvl], [0, data.shape[1]], color='white', alpha=0.5) 
                    elif key is 'jac':
                        for i in range(1, len(grid)):
                            lvl = i * grid[i]
                            ax.plot([0, data.shape[0]], [lvl, lvl], color='white', alpha=0.5)
                    fig.tight_layout()
                    if self.opts.params and key == 'jac':
                        lines = '\n'
                        lines = lines.join(self.data[gname]['params'])
                        plot_lines(lines, loc='upper right', outside=[-.42, 1.02])
                    if self.opts.out_fig:
                        tmp_path = path[0]+'_{:02}_{}_{}.{}'.format(num, gname, key, path[1])
                        plt.savefig(tmp_path)
                        num += 1
                        print('Save output file: '+tmp_path)
                    if self.opts.show:
                        plt.show()
                    fig.clf()
                    fig.clear()
                    plt.close()




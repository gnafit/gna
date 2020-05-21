# -*- coding: utf-8 -*-
from __future__ import print_function
from gna.ui import basecmd
import numpy as np
import argparse
import os
from os.path import join, abspath
from pkgutil import iter_modules
from gna.config import cfg
from itertools import product
import runpy
import sys
from gna.env import env
from collections import defaultdict
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('experiments', nargs=2, help='Models to compare')
        parser.add_argument('--mapping', default=None, help='Mappings between names of observables')
        parser.add_argument('--ns', help='namespace')
        parser.add_argument('--output', help='Path where to save pics')
        parser.add_argument('-e', '--embed', action='store_true')
        parser.add_argument('-s', '--show', action='store_true', help='Show pics')
        parser.add_argument('-x', '--xaxis', type=lambda x: env.get(x), help='Observable to use as x axis' )

    def __init__(self, *args, **kwargs):
        basecmd.__init__(self, *args, **kwargs)
        self.observables = defaultdict(lambda: defaultdict(lambda: defaultdict()))
        if self.opts.mapping:
            self.load_mapping()

    def init(self):
        self.check_exps_correct()
        self.extract_observables()
        if self.opts.embed:
            import IPython; IPython.embed()
        self.make_plots()

    def make_plots(self):
        for title, obses in self.observables.items():
            if self.opts.output:
                pp = PdfPages(str(abspath( join(self.opts.output, title) ) + '.pdf'))

            #  import IPython; IPython.embed()
            for obs_name, obs_pair in obses.items():
                fig, (ax, ax_for_ratio) = plt.subplots(nrows=2, ncols=1, sharex=True)
                fig.subplots_adjust(hspace=0.05)
                fig.set_tight_layout(True)

                for exp, obs in obs_pair.items():
                    if self.opts.xaxis:
                        ax.plot(self.opts.xaxis.data(), obs, label=' '.join((obs_name, exp)))
                    else:
                        ax.plot(obs, label=label)

                ax.set_title(title, fontsize=16)
                ax.set_xlabel(self.opts.xaxis.label())
                ax.legend(loc='best')

                first_exp, second_exp = self.exps
                #  print(obs_pair.keys())
                ratio = obs_pair[first_exp] / obs_pair[second_exp]
                if self.opts.xaxis:
                    ax_for_ratio.plot(self.opts.xaxis.data(), ratio, label='/'.join(self.exps))
                else:
                    ax_for_ratio.plot(ratio, label='/'.join(self.exps))
                ax_for_ratio.legend(loc='best')
                ax_for_ratio.set_title('Ratio', fontsize=16)
                sanitized = ratio[np.isfinite(ratio)]
                #  high = np.percentile(sanitized, 95, interpolation='linear')
                #  low  = np.percentile(sanitized, 5, interpolation='linear')
                #  interval = high - low
                #  ax_for_ratio.set_ylim((low - interval*5, high + 5*interval))
                ax_for_ratio.axhline(y=1.0, linestyle='--', color='grey', alpha=0.5)

                if self.opts.show:
                    plt.show()

                if self.opts.output:
                    pp.savefig(fig)
                    plt.close('all')

            if self.opts.output:
                pp.close()


    def extract_observables(self):
        for obs_pair, label in zip(self.mapping['observables'], self.mapping['labels']):
                for idx_pair in self.mapping['indices']:
                    obs_common_name = obs_pair[0].format(idx_pair[0])
                    try:
                        label_formated = label.format(idx_pair[0])
                    except:
                        label_formated = label
                    for obs_template, idx, exp in zip(obs_pair, idx_pair, self.exps):
                        obs = obs_template.format(idx)
                        self.observables[label_formated][obs_common_name][exp] = env.get('{0}/{1}'.format(exp, obs)).data()
                        print(label_formated, obs, exp)

    def load_mapping(self):
        map_path = os.path.abspath(self.opts.mapping)
        loaded = runpy.run_path(map_path)
        self.exps, self.mapping = loaded['exps'], loaded['mapping']

    def check_exps_correct(self):
        if self.exps != tuple(self.opts.experiments):
                reverse = tuple(reversed(self.opts.experiments))
                if reverse != self.exps:
                    raise Exception("Experiments from cli {0} and mapping "
                                    "config {1} doesn't match".format(self.opts.experiments, self.exps))
                self.exps = reverse

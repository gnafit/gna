#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt
import os
import pytest
from pathlib import Path

from load import ROOT as R

from mpl_tools.helpers import savefig
from gna.configurator import NestedDict
from gna.bundle import execute_bundles
from gna.unittest import allure_attach_file, savegraph
from gna.env import env
from gna.expression.expression_v01 import Expression_v01, ExpressionContext_v01
from gna.expression.index import NIndex
import gna.constructors as C


def test_oscprob_msw_approx_v01(tmp_path):

    _enu = np.linspace(1.8, 10.0, 500, dtype='d')
    Enu = C.Points(_enu, labels='anue energy')

    indices = [
         ('d', 'detector', ['AD1']),
         ('r', 'reactor', ['YJ1', 'YJ2']),
         ('c', 'component', ['comp0', 'comp12', 'comp13', 'comp23'])
        ]

    fake_data_dir = Path(__file__).resolve().parent / "msw_approx"
    fake_reactors = fake_data_dir / "fake_baselines.yaml"


    expr = ['baseline[d,r]',
            'oscprob_msw_approx[d,r]|enu()',
            'vacuum_oscprob = sum[c]| pmns[c]*oscprob[c,d,r](enu())',
            'oscprob_matter[d,r](enu())'
            ]
    a =  Expression_v01(expr, indices = NIndex.fromlist(indices))
    a.parse()
    print(a.expressions)
    lib = dict()
    a.guessname(lib, save=True)

    ns_anuexpr = env.globalns('anue_expr')
    cfg = NestedDict(
            baselines = dict(
                    bundle = dict(name='reactor_baselines', version='v02', major = 'rd'),
                    reactors  = fake_reactors.as_posix(),
                    reactors_key = 'reactor_baseline',
                    detectors = dict(AD1=0.0),
                    unit = 'km'
                    ),
            enu = NestedDict(
                bundle = NestedDict(name='predefined', version='v01', major=''),
                name = 'enu',
                inputs = None,
                outputs = Enu.single(),
                ),
            approx_oscprob = dict(bundle = dict(name='oscprob_approx', version='v01', major='rd'),
                    parameters = dict(
                        DeltaMSq23    = 2.453e-03,
                        DeltaMSq12    = 7.53e-05,
                        SinSqDouble13 = (0.08529904, 0.00267792),
                        SinSqDouble12 = 0.851004,
                        ),
                    density=2.6,
                    ),
            vacuum_oscprob = dict( bundle = dict(name='oscprob', version='v05', major='rdc'),
                    parameters = dict(
                        DeltaMSq23    = 2.453e-03,
                        DeltaMSq12    = 7.53e-05,
                        SinSqDouble13 = (0.08529904, 0.00267792),
                        SinSqDouble12 = 0.851004,
                        )
                    ),
                oscprob_matter = dict(
                    bundle = dict(name='oscprob_matter', version='v01', major='rd',
                                  names=dict(oscprob='oscprob_matter')),
                    density = 2.6, # g/cm3
                    dm      = '23',
                    skip_conf=True,
                    ),
            )
    context = ExpressionContext_v01(cfg, ns=ns_anuexpr)
    a.build(context)
    ns_anuexpr.printparameters(labels=True)

    oscprob_approx_msw = context.outputs.oscprob_msw_approx.AD1.YJ1.data().copy()
    oscprob_vacuum_no = context.outputs.vacuum_oscprob.AD1.YJ1.data().copy()
    oscprob_matter = context.outputs.oscprob_matter.AD1.YJ1.data().copy()

    rho = ns_anuexpr['rho']
    alpha = ns_anuexpr['pmns.Alpha']
    rho.push(0)
    oscprob_approx_msw_vacuum_no = context.outputs.oscprob_msw_approx.AD1.YJ1.data().copy()
    oscprob_matter_vacuum_no = context.outputs.oscprob_matter.AD1.YJ1.data().copy()

    alpha.push('inverted')
    oscprob_vacuum_io = context.outputs.vacuum_oscprob.AD1.YJ1.data().copy()
    oscprob_approx_msw_vacuum_io = context.outputs.oscprob_msw_approx.AD1.YJ1.data().copy()

    alpha.pop()
    rho.pop()

    ratio_vacuum_no = oscprob_approx_msw_vacuum_no/oscprob_vacuum_no
    ratio_vacuum_io = oscprob_approx_msw_vacuum_io/oscprob_vacuum_io
    ratio_to_exact_vacuum  = oscprob_approx_msw_vacuum_no / oscprob_matter_vacuum_no
    ratio_to_exact  = oscprob_approx_msw / oscprob_matter

    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True, gridspec_kw=dict(hspace=0.))
    axes[0].plot(_enu, oscprob_approx_msw, alpha=0.9, label='Approximate MSW NO')
    axes[0].plot(_enu, oscprob_matter, alpha=0.5, label='Exact MSW NO')
    axes[0].plot(_enu, oscprob_vacuum_no, alpha=0.5, label='Vacuum NO')
    axes[0].plot(_enu, oscprob_vacuum_io, alpha=0.5, label='Vacuum IO')
    axes[0].legend(loc='best')

    axes[1].plot(_enu, np.log(ratio_vacuum_no), label='NO')
    axes[1].plot(_enu, np.log(ratio_vacuum_io), label='IO')
    axes[1].plot(_enu, np.log(ratio_to_exact_vacuum), label='Exact')
    axes[1].legend(loc='best', ncol=3, title='Vacuum log-ratio')
    offset = 0.3e-14
    axes[1].set_ylim(-offset, +offset)

    axes[2].plot(_enu, np.log(ratio_to_exact), label='MSW log-ratio approx/exact')
    axes[2].legend(loc='best')

    suffix = 'matter_oscprob'
    path = os.path.join(str(tmp_path), suffix + '.pdf')
    savefig(path, dpi=300)
    #  allure_attach_file(path)

    path = os.path.join(str(tmp_path), suffix + '_graph.pdf')
    savegraph(context.outputs.oscprob_msw_approx.AD1.YJ1, path)
    allure_attach_file(path)

    plt.show()

    #  #  savefig(path.replace('.png','.pdf'), dpi=300)

    #  path = os.path.join(str(tmp_path), suffix + '_graph.png')
    #  savegraph(u235_spec, path)
    #  allure_attach_file(path)
    #  path = os.path.join(str(tmp_path), suffix + '_graph.pdf')
    #  savegraph(u235_spec, path)
    #  allure_attach_file(path)

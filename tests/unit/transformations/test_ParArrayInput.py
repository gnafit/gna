#!/usr/bin/env python

import load
from gna.env import env
import gna.constructors as C
import numpy as np
from gna.unittest import allure_attach_file, savegraph
import os

def test_ParArrayInput_v01(tmp_path):
    ns = env.globalns("test_ParArrayInput_v01")

    size=10
    values_initial = np.ones(size, dtype='d')
    values_new = np.arange(size, dtype='d')+2
    names = [f'par_{i:02d}' for i in range(size)]

    pars = [
            ns.defparameter(name, central=value, relsigma=0.1)
            for name, value in zip(names, values_initial)
            ]

    with ns:
        Vars = C.VarArray(names, labels='All variables')

    Pars = C.ParArrayInput(labels='Parameter inputs')
    Pars.add_input()
    for par in pars[::2]:
        Pars.append(par)

    Pars.add_input()
    for par in pars[1::2]:
        Pars.append(par)
    Pars.materialize()

    Values_new1 = C.Points(values_new[::2], labels='Input array 1 (odd)')
    Values_new2 = C.Points(values_new[1::2], labels='Input array 2 (even)')
    Values_new1 >> Pars.pararray.points
    Values_new2 >> Pars.pararray.points_02

    Pars.printtransformations()
    ns.printparameters()

    res0=Vars.vararray.points.data()
    assert np.allclose(values_initial, res0, atol=0, rtol=0)

    Pars.pararray.touch()
    res1=Vars.vararray.points.data()
    assert np.allclose(values_new, res1, atol=0, rtol=0)

    ns.printparameters()

    path = os.path.join(str(tmp_path), 'pararrayinput.png')
    savegraph(Pars.pararray, path, namespace=ns)
    allure_attach_file(path)

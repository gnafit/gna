#!/usr/bin/env python

from load import ROOT as R
import gna.constructors as C
import numpy as np
from gna.unittest import *
from gna.env import env

@passname
def test_switch(function_name):
    ninputs = 4
    ones = np.ones((3,4))
    objects = [C.Points(i*ones) for i in range(ninputs)]

    ns = env.globalns(function_name)
    switchvar = ns.defparameter('switch', central=1.0, fixed=True)
    with ns:
        switch = C.Switch('switch', outputs=[p.points.points for p in objects])

    trans = switch.switch
    result = trans.result

    assert trans.tainted()
    assert (result()==1.0).all()
    assert not trans.tainted()

    for i in range(ninputs):
        switchvar.set(i)
        assert trans.tainted()
        assert (result()==i).all()
        assert not trans.tainted()

    objects[-1].points.taint()
    assert trans.tainted()

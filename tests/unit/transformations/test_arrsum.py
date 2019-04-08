#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
from load import ROOT as R
from gna.env import env
from gna.parameters.printer import print_parameters
from gna import constructors as C
from gna.unittest import *

@floatcopy(globals(), addname=True)
def test_arrsum(function_name):
    varname = 'out'
    ns = env.globalns(function_name)
    names = ["var1", "var2", "var3", "var4"]
    variables = [ns.reqparameter(name, central=float(i), relsigma=0.1)
                for i, name in enumerate(names)]

    with ns:
        var_arr = C.VarArray(names)
    print("Input var array ", var_arr.vararray.points.data())

    sum_arr = C.ArraySum(varname, var_arr, ns=ns)
    # materialize variable
    ns[varname].get()
    output = var_arr.vararray.points
    print('Data:', output.data(), output.data().sum())
    print("Value of %s evaluable immediately after initialization "%varname, ns[varname].value(), sum_arr.arrsum.sum.data())
    print()
    assert (output.data().sum()==ns[varname].value()).all()

    #  sum_arr.arrsum.arr(var_arr.vararray)
    #  sum_arr.exposeEvaluable(var_arr.vararray)
    #  print(sum_arr.arrsum.accumulated.data())
    print("Change value of var1 variable to 10")
    ns['var1'].set(10)
    print('Data:', output.data(), output.data().sum())
    ns[varname].dump()
    print("Sum should now be ", np.sum(var_arr.vararray.points.data()))
    print("Check the value %s of evaluable now: "%varname, ns['out'].value(), sum_arr.arrsum.sum.data())
    assert (output.data().sum()==ns[varname].value()).all()
    print()

    ns.printparameters()

if __name__ == "__main__":
    run_unittests(globals())

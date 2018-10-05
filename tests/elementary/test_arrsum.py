#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
from load import ROOT as R
from gna.env import env
from gna.parameters.printer import print_parameters
from converters import convert

ns = env.globalns
names = ["var1", "var2", "var3", "var4"]
variables = [ns.reqparameter(name, central=float(i), relsigma=0.1)
            for i, name in enumerate(names)]

cpp_names = convert(names, 'stdvector')
var_arr = R.VarArray(cpp_names)
print("Input var array ", var_arr.vararray.points.data())

sum_arr = R.ArraySum(var_arr, ns=ns)
ns['out'].get()
output = var_arr.vararray.points
print('Data:', output.data(), output.data().sum())
print("Value of \"out\" evaluable immediately after initialization ", ns['out'].value())
#  sum_arr.arrsum.arr(var_arr.vararray)
#  sum_arr.exposeEvaluable(var_arr.vararray)
#  print(sum_arr.arrsum.accumulated.data())
print("Change value of var1 variable to 10")
ns['var1'].set(10)
print('Data:', output.data(), output.data().sum())
ns['out'].dump()
print("Sum should now be ", np.sum(var_arr.vararray.points.data()))

print("Check the value \"out\" of evaluable now", ns['out'].value())

env.globalns.printparameters()




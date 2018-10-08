#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import numpy as np
from load import ROOT as R
from gna.env import env
from gna.parameters.printer import print_parameters
from converters import convert

varname = 'out'
ns = env.globalns
names = ["var1", "var2", "var3", "var4"]
variables = [ns.reqparameter(name, central=float(i), relsigma=0.1)
            for i, name in enumerate(names)]

cpp_names = convert(names, 'stdvector')
var_arr = R.VarArray(cpp_names)
print("Input var array ", var_arr.vararray.points.data())

sum_arr = R.ArraySum(varname, var_arr, ns=ns)
ns[varname].get()
output = var_arr.vararray.points
print('Data:', output.data(), output.data().sum())
print("Value of %s evaluable immediately after initialization "%varname, ns[varname].value())
#  sum_arr.arrsum.arr(var_arr.vararray)
#  sum_arr.exposeEvaluable(var_arr.vararray)
#  print(sum_arr.arrsum.accumulated.data())
print("Change value of var1 variable to 10")
ns['var1'].set(10)
print('Data:', output.data(), output.data().sum())
ns[varname].dump()
print("Sum should now be ", np.sum(var_arr.vararray.points.data()))

print("Check the value %s of evaluable now"%varname, ns['out'].value())

env.globalns.printparameters()




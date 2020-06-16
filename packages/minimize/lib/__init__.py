# -*- coding: utf-8 -*-
from packages.minimize.lib import minuit
from packages.minimize.lib import scipyminimizer

minimizers = {
    'minuit': minuit.Minuit,
    'scipy': scipyminimizer.SciPyMinimizer,
}

try:
    from packages.minimize.lib import minuit2
    minimizers['minuit2'] = minuit2.Minuit2
except ImportError:
    pass

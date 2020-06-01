from packages.minimize.lib import minuit
from packages.minimize.lib import scipyminimizer
# from packages.minimize.lib import minuit2

minimizers = {
    'minuit': minuit.Minuit,
    # 'minuit2': minuit2.Minuit2
    'scipy': scipyminimizer.SciPyMinimizer,
}


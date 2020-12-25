from packages.minimize.lib import scipyminimizer

minimizers = {
    'scipy': scipyminimizer.SciPyMinimizer,
}

try:
    from packages.minimize.lib import minuit
    minimizers['minuit'] = minuit.Minuit
except Exception:
    print("\033[31mUnable to load Minuit minimizer from ROOT.\033[0m")

try:
    from packages.minimize.lib import minuit2
    minimizers['minuit2'] = minuit2.Minuit2
except Exception:
    print("\033[31mUnable to load Minuit2 minimizer from ROOT.\033[0m")


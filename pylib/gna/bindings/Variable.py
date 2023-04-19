from gna.bindings import patchROOTClass, provided_precisions
import ROOT as R

classes = []
for ft in provided_precisions:
    classes.append(R.Variable(ft))

@patchROOTClass(classes, 'materialize')
def Variable__materialize(self, attr):
    return self

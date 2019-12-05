#!/usr/bin/env python

from __future__ import print_function
from tutorial import tutorial_image_name, savefig
import load
import gna.constructors as C
from matplotlib import pyplot as plt
from gna.bindings import common
import ROOT as R
import itertools as I





# Create ROOT objects
roothist1d = R.TH1D('testhist', 'testhist', 10, -5, 5)
roothist2d = R.TH2D('testhist', 'testhist', 20, 0, 10, 24, 0, 12)
rootmatrix = R.TMatrixD(3,4)

# Fill objects
# Fill TH1D
roothist1d.FillRandom('gaus', 10000)

# Fill TH2D with user defined function
xyg=R.TF2("xyg","exp([0]*x)*exp([1]*y)", 0, 10, 0, 12)
xyg.SetParameter(0, -1/2.)
xyg.SetParameter(1, -1/8.)
R.gDirectory.Add( xyg )
roothist2d.FillRandom('xyg', 10000)

# Fill TMatrixD with
for i, (i1, i2) in enumerate(I.product(range(rootmatrix.GetNrows()), range(rootmatrix.GetNcols()))):
    rootmatrix[i1, i2] = i

# Create Points
p1 = C.Points(roothist1d)
p2 = C.Points(roothist2d)
p3 = C.Points(rootmatrix)

# Create Histograms
h1d = C.Histogram(roothist1d)
h2d = C.Histogram2d(roothist2d)

# Check p1
print('Points from TH1D (underflow/overflow are ignored)')

roothist1d.Print('all')
print(p1.points.points())

fig = plt.figure()
ax = plt.subplot( 111 )
ax.minorticks_on()
ax.grid()
ax.set_xlabel( 'X axis' )
ax.set_ylabel( 'Entries' )
ax.set_title( 'Points and TH1 comparison' )

roothist1d.plot(alpha=0.6, linestyle='dashdot', label='TH1')
h1d.hist.hist.plot_hist(alpha=0.6, linestyle='dashed', label='Histogram')
p1.points.points.plot_hist(label='Points')
ax.legend(loc='upper center', ncol=3)

ax.set_ylim(top=ax.get_ylim()[1]*1.1)

savefig(tutorial_image_name('png', suffix='1d'))

# Check p2
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(8, 8),
                                             gridspec_kw=dict(hspace=0.3, wspace=0.35, left=0.10, right=0.90))
ax1.minorticks_on()
ax1.grid()
ax1.set_xlabel( 'X axis' )
ax1.set_ylabel( 'Y axis' )
ax1.set_title( 'TH2' )

plt.sca(ax1)
h2d.hist.hist.plot_pcolorfast(colorbar=True)

ax2.minorticks_on()
ax2.grid()
ax2.set_xlabel( 'X axis' )
ax2.set_ylabel( 'Y axis' )
ax2.set_title( 'Histogram2d' )

plt.sca(ax2)
roothist2d.pcolorfast(colorbar=True)

ax3.minorticks_on()
ax3.grid()
ax3.set_xlabel( 'X axis' )
ax3.set_ylabel( 'Y axis' )
ax3.set_title( 'Points 2d' )

plt.sca(ax3)
p2.points.points.plot_pcolorfast(colorbar=True)

plt.sca(ax4)
plt.axis('off')

savefig(tutorial_image_name('png', suffix='2d'))


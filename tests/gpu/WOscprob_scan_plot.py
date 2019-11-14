#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from h5py import File
from matplotlib import pyplot as P
from matplotlib.backends.backend_pdf import PdfPages
from gna.labelfmt import formatter as L

from argparse import ArgumentParser
parser = ArgumentParser()
parser.add_argument('input', type=lambda fname: File(fname, 'r'))
parser.add_argument('-o', '--output', type=PdfPages)
parser.add_argument('-s', '--show', action='store_true')
args = parser.parse_args()
data = args.input

def savefig():
    if not args.output:
        return

    args.output.savefig()

first_element=0
last_element=None
size = data['size'][first_element:last_element]
opts = dict( markerfacecolor='none' )

def set_scale(scale='log', relative=False):
    ax = P.gca()
    if scale=='log':
        ax.set_xscale('log')
        if not relative:
            ax.set_yscale('log')
    else:
        ax.set_ylim(bottom=0.0)
        if relative:
            ax.set_xlim(top=1.05)

fig = P.figure()
ax = P.subplot( 111 )
ax.minorticks_on()
ax.grid()
ax.set_xlabel('Input size')
ax.set_ylabel('Time per execution, s')
ax.set_title('CPU double')

ax.plot( size, data['cpu']['double']['all'][first_element:last_element],        'o-', label='all',                       **opts )
ax.plot( size, data['cpu']['double']['SinSq13'][first_element:last_element],    'o-', label=r'$\theta_{13}$',            **opts )
ax.plot( size, data['cpu']['double']['DeltaMSqEE'][first_element:last_element], 'o-', label=r'$\Delta m^2_\mathrm{ee}$', **opts )
ax.legend(title='Modification:')
savefig()
set_scale()
savefig()

fig = P.figure()
ax = P.subplot( 111 )
ax.minorticks_on()
ax.grid()
ax.set_xlabel('Input size')
ax.set_ylabel('Ratio')
ax.set_title('CPU double, ratio')

ref = data['cpu']['double']['all'][first_element:last_element]
ax.plot( size, data['cpu']['double']['all'][first_element:last_element]/ref,        'o-', label='all',                       **opts )
ax.plot( size, data['cpu']['double']['SinSq13'][first_element:last_element]/ref,    'o-', label=r'$\theta_{13}$',            **opts )
ax.plot( size, data['cpu']['double']['DeltaMSqEE'][first_element:last_element]/ref, 'o-', label=r'$\Delta m^2_\mathrm{ee}$', **opts )
ax.legend(title='Modification:')
set_scale(relative=True)
savefig()


fig = P.figure()
ax = P.subplot( 111 )
ax.minorticks_on()
ax.grid()
ax.set_xlabel('Input size')
ax.set_ylabel('Time per execution, s')
ax.set_title('CPU')

ld = ax.plot( size, data['cpu']['double']['all'][first_element:last_element], 'o-', label='double', **opts )
ax.plot( size, data['cpu']['double']['SinSq13'][first_element:last_element], 'o--', color=ld[0].get_color(), label='double (mixing angle)', **opts )
lf = ax.plot( size, data['cpu']['float']['all'][first_element:last_element],  'o-', label='float',  **opts )
ax.plot( size, data['cpu']['float']['SinSq13'][first_element:last_element],  'o--', color=lf[0].get_color(), label='float (mixing angle)',  **opts )
ax.legend(title='Float precision:')
savefig()
set_scale()
savefig()


fig = P.figure()
ax = P.subplot( 111 )
ax.minorticks_on()
ax.grid()
ax.set_xlabel('Input size')
ax.set_ylabel('Ratio')
ax.set_title('CPU, ratio to double')

refa = data['cpu']['double']['all'][first_element:last_element]
refs = data['cpu']['double']['SinSq13'][first_element:last_element]
# ld = ax.plot( size, data['cpu']['double']['all'][first_element:last_element]/refa, 'o-', label='double', **opts )
ax.plot( size, data['cpu']['float']['all'][first_element:last_element]/refa,  'o-', label='all',  **opts )
# ax.plot( size, data['cpu']['double']['SinSq13'][first_element:last_element]/refd, 'o--', color=ld[0].get_color(), label='double (mixing angle)', **opts )
ax.plot( size, data['cpu']['float']['SinSq13'][first_element:last_element]/refs,  'o-', label=r'$\theta_{13}$',  **opts )
ax.legend(title='Float precision:')
set_scale(relative=True)
savefig()


fig = P.figure()
ax = P.subplot( 111 )
ax.minorticks_on()
ax.grid()
ax.set_xlabel('Input size')
ax.set_ylabel('Time per execution, s')
ax.set_title('CPU and GPU')

lcpu = ax.plot( size, data['cpu']['double']['all'][first_element:last_element], 'o-', label='CPU', **opts )[0]
ax.plot( size, data['cpu']['float']['all'][first_element:last_element], 'o--', color=lcpu.get_color(), label='CPU (float)', **opts )
lgpu = ax.plot( size, data['gpu']['double']['all'][first_element:last_element], 'o-', label='GPU',  **opts )[0]
ax.plot( size, data['gpu']['float']['all'][first_element:last_element], 'o--', color=lgpu.get_color(), label='GPU (float)', **opts )
ax.legend(title='Device:')
savefig()
set_scale()
savefig()


fig = P.figure()
ax = P.subplot( 111 )
ax.minorticks_on()
ax.grid()
ax.set_xlabel('Input size')
ax.set_ylabel('Ratio')
ax.set_title('CPU and GPU, ratio to double')

refcpu = data['cpu']['double']['all'][first_element:last_element]
refgpu = data['gpu']['double']['all'][first_element:last_element]
ax.plot( size, data['cpu']['float']['all'][first_element:last_element]/refcpu, 'o-', label='CPU (float)', **opts )
ax.plot( size, data['gpu']['float']['all'][first_element:last_element]/refgpu, 'o-', label='GPU (float)', **opts )
ax.legend(title='Device:')
set_scale(relative=True)
savefig()

fig = P.figure()
ax = P.subplot( 111 )
ax.minorticks_on()
ax.grid()
ax.set_xlabel('Input size')
ax.set_ylabel('Time per execution, s')
ax.set_title('CPU and GPU, ratio to GPU')

refdouble = data['gpu']['double']['all'][first_element:last_element]
reffloat = data['gpu']['float']['all'][first_element:last_element]
ax.plot( size, data['cpu']['double']['all'][first_element:last_element]/refdouble, 'o-', label='CPU double', **opts )
ax.plot( size, data['cpu']['float']['all'][first_element:last_element]/reffloat, 'o-', label='CPU float', **opts )
ax.legend(title='Precision:')

set_scale(relative=True)
savefig()

fig = P.figure()
ax = P.subplot( 111 )
ax.minorticks_on()
ax.grid()
ax.set_xlabel('Input size')
ax.set_ylabel('Ratio')
ax.set_title(r'CPU and GPU, partial modification ($\theta_{13}$), ratio to double')

refcpu = data['cpu']['double']['SinSq13'][first_element:last_element]
refgpu = data['gpu']['double']['SinSq13'][first_element:last_element]
ax.plot( size, data['cpu']['float']['SinSq13'][first_element:last_element]/refcpu, 'o-', label='CPU (float)', **opts )
ax.plot( size, data['gpu']['float']['SinSq13'][first_element:last_element]/refgpu, 'o-', label='GPU (float)', **opts )
ax.legend(title='Device:')
set_scale(relative=True)
savefig()

fig = P.figure()
ax = P.subplot( 111 )
ax.minorticks_on()
ax.grid()
ax.set_xlabel('Input size')
ax.set_ylabel('Time per execution, s')
ax.set_title(r'CPU and GPU, partial modification ($\theta_{13}$), ratio to GPU')

refdouble = data['gpu']['double']['SinSq13'][first_element:last_element]
reffloat = data['gpu']['float']['SinSq13'][first_element:last_element]
ax.plot( size, data['cpu']['double']['SinSq13'][first_element:last_element]/refdouble, 'o-', label='CPU double', **opts )
ax.plot( size, data['cpu']['float']['SinSq13'][first_element:last_element]/reffloat, 'o-', label='CPU float', **opts )
ax.legend(title='Precision:')

set_scale(relative=True)
savefig()

fig = P.figure()
ax = P.subplot( 111 )
ax.minorticks_on()
ax.grid()
ax.set_xlabel( L.u('enu') )
ax.set_ylabel( L.u('psur') )
ax.set_title( 'Survival probability' )

stride = 1
ax.plot( data['cpu']['double']['psur'][:][::stride], '-', label='CPU double', alpha=0.5, **opts )
ax.plot( data['cpu']['float']['psur'][:][::stride], '-',  label='CPU float', alpha=0.5, **opts )
ax.plot( data['gpu']['double']['psur'][:][::stride], '-', label='GPU double', alpha=0.5, **opts )
ax.plot( data['gpu']['float']['psur'][:][::stride], '-',  label='GPU float', alpha=0.5, **opts )

ax.legend()
savefig()

fig = P.figure()
ax = P.subplot( 111 )
ax.minorticks_on()
ax.grid()
ax.set_xlabel( L.u('enu') )
ax.set_ylabel( L.u('psur') )
ax.set_title( 'Survival probability' )

ax.plot( data['cpu']['double']['psur'][:][::stride], '-', label='CPU double', **opts )
savefig()

fig = P.figure()
ax = P.subplot( 111 )
ax.minorticks_on()
ax.grid()
ax.set_xlabel( L.u('enu') )
ax.set_ylabel( '$\\mathrm{Ratio}-1$' )
ax.set_title( 'Survival probability, ratio to CPU double' )

ref = data['cpu']['double']['psur'][:][::stride]
# ax.plot( data['cpu']['double']['psur'][:][::stride]/ref-1, '-', label='CPU double', alpha=0.5, **opts )
ax.plot( data['cpu']['float']['psur'][:][::stride]/ref-1, '-',  label='CPU float', alpha=0.5, **opts )
ax.plot( data['gpu']['float']['psur'][:][::stride]/ref-1, '-',  label='GPU float', alpha=0.5, **opts )
ax.yaxis.get_major_formatter().set_powerlimits((-3,3))

ax.legend(title='Device:')
savefig()

fig = P.figure()
ax = P.subplot( 111 )
ax.minorticks_on()
ax.grid()
ax.set_xlabel( L.u('enu') )
ax.set_ylabel( '$\\mathrm{Ratio}-1$' )
ax.set_title( 'Survival probability, ratio to CPU double' )

ref = data['cpu']['double']['psur'][:][::stride]
# ax.plot( data['cpu']['double']['psur'][:][::stride]/ref-1, '-', label='CPU double', alpha=0.5, **opts )
ax.plot( data['gpu']['double']['psur'][:][::stride]/ref-1, '-', label='GPU double', alpha=0.5, **opts )

ax.legend()
savefig()

if args.show:
    P.show()

if args.output:
    args.output.close()


#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from gna.configurator import configurator, pprint
from collections import OrderedDict
from physlib import percent

cfg = configurator()
cfg.detectors = [ 'AD11', 'AD12', 'AD21', 'AD22', 'AD31', 'AD32', 'AD33', 'AD34' ]
cfg.groups    = OrderedDict( [
                ( 'EH1', ('AD11', 'AD12') ),
                ( 'EH2', ('AD21', 'AD22') ),
                ( 'EH3', ('AD31', 'AD32', 'AD33', 'AD34') ),
                ] )

bkg = cfg('bkg')
bkg.list = [ 'acc', 'lihe', 'fastn', 'amc', 'alphan' ]

acc = cfg.bkg('acc')
acc.uncertainty      = 1.0*percent
acc.uncertainty_mode = 'relative'

lihe = cfg.bkg('lihe')
lihe.rates = configurator(
            lifraction = 0.95,
            lifraction_unc = 5.0*percent,
        )

    # lihe       = dict( EH1=0.90,  EH2=0.73,  EH3=0.07,  mode='absolute' ),
    # mode       = 'relative',
    # alphan     = 50.0*percent,
    # fastn      = dict( EH1=0.083, EH2=0.062, EH3=0.009, mode='absolute' ),
    # fastn_ihep = dict( EH1=0.103, EH2=0.074, EH3=0.009, mode='absolute' ),
    # lihe_ihep  = dict( EH1=1.06,  EH2=0.77,  EH3=0.06,  mode='absolute' ),
    # amc        = dict( AD11=0.08, AD12=0.08,
                       # AD21=0.07, AD22=0.07,
                       # AD31=0.03, AD32=0.03, AD33=0.03, AD34=0.02,
                       # mode='absolute'
                       # ),
    # amc6ad     = dict( AD11=0.12, AD12=0.11,
                       # AD21=0.13, #AD22=0.00,
                       # AD31=0.10, AD32=0.10, AD33=0.10, #AD34=0.00,
                       # mode='absolute'
                       # ),
    # amc8ad     = dict( AD11=0.07, AD12=0.06,
                       # AD21=0.06, AD22=0.07,
                       # AD31=0.02, AD32=0.02, AD33=0.02, AD34=0.02,
                       # mode='absolute'
                       # ),
    # amcp14a8ad = dict( AD11=0.09, AD12=0.10,
                       # AD21=0.08, AD22=0.10,
                       # AD31=0.03, AD32=0.02, AD33=0.02, AD34=0.03,
                       # mode='absolute'
                       # ),
    # amcp14a    = dict( AD11=0.10, AD12=0.10,
                       # AD21=0.09, AD22=0.10,
                       # AD31=0.05, AD32=0.04, AD33=0.04, AD34=0.03,
                       # mode='absolute'
                       # ),

pprint( cfg )
import IPython
IPython.embed()

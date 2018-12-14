# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import numpy as N
import gna.constructors as C
from gna.bundle import *
from collections import OrderedDict
import h5py as H

class dayabay_livetime_hdf_v01(TransformationBundle):
    def __init__(self, **kwargs):
        super(dayabay_livetime_hdf_v01, self).__init__( **kwargs )

        self.init_indices()
        if not self.idx.ndim()==1:
            raise self.exception('require 1 index exactly')
        self.init_data()

    def init_data(self):
        self.data = OrderedDict()
        with H.File(self.cfg.file, 'r') as f:
            self.info = f['info'][0]

            byday = f['byday']
            for key, addata in byday.items():
                key = 'AD'+key
                self.data[key] = addata[:]

    def define_variables(self):
        daq = self.common_namespace('daq')
        daq.reqparameter('first_day', central=self.info['first_day'], fixed=True, sigma=0.01,
                          label='First DAQ day start: %s'%self.info['str_first_day'] )
        daq.reqparameter('last_day', central=self.info['last_day'], fixed=True, sigma=0.01,
                          label='Last DAQ day end:    %s'%self.info['str_last_day'] )
        daq.reqparameter('ndays', central=self.info['days'], fixed=True, sigma=0.01,
                          label='Total number of days' )

        data_lt = 0.0
        for it in self.idx.iterate():
            ad, = it.current_values()
            data = self.data.get(ad, None)
            if data is None:
                raise self.exception('Failed to retrieve data for %s from %s'%(ad, self.cfg.file))

            data_lt = data['livetime']+data_lt
            livetime    = C.Points(data['livetime'])
            livetime.points.setLabel(it.current_format('Livetime\n{autoindex}'))
            eff         = C.Points(data['eff'])
            eff.points.setLabel(it.current_format('Efficiency (mu*mult)\n{autoindex}'))
            efflivetime = R.Product(livetime, eff)
            efflivetime.product.setLabel(it.current_format('Livetime (eff)\n{autoindex}'))

            self.objects[('livetime',ad)] = livetime
            self.objects[('eff',ad)] = eff
            self.objects[('efflivetime',ad)] = efflivetime

            self.set_output(livetime.single(), 'livetime_daily', it)
            self.set_output(eff.single(), 'eff_daily', it)
            self.set_output(efflivetime.single(), 'efflivetime_daily', it)

        ndays_daq = (data_lt>0.0).sum()
        daq.reqparameter('ndays_daq', central=ndays_daq, fixed=True, sigma=0.01,
                          label='Total number of DAQ days (exclude no DAQ)' )

    def varname(self, var):
        mapping = self.cfg.get('names', None)
        return mapping.get(var, var) if mapping else var


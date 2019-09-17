# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT as R
import numpy as N
import gna.constructors as C
from gna.bundle import TransformationBundle
from collections import OrderedDict
import h5py as H

class dayabay_livetime_hdf_v02(TransformationBundle):
    def __init__(self, *args, **kwargs):
        TransformationBundle.__init__(self, *args, **kwargs)
        self.check_nidx_dim(1, 1, 'major')
        self.check_nidx_dim(0, 0, 'minor')

        self.init_data()

    @staticmethod
    def _provides(cfg):
        return ('first_day', 'last_day', 'ndays', 'ndays_daq', 'seconds_in_day', 'days_in_second'), \
               ('eff_daily', 'livetime_daily', 'efflivetime_daily')

    def init_data(self):
        self.data = OrderedDict()
        with H.File(self.cfg.file, 'r') as f:
            self.info = f['info'][0]

            byday = f['byday']
            for key, ad_data in byday.items():
                key = 'AD'+key
                self.data[key] = ad_data[:]

    def define_variables(self):
        daq = self.namespace('daq')
        daq.reqparameter('first_day', central=self.info['first_day'], fixed=True,
                          label='First DAQ day start: %s'%self.info['str_first_day'] )
        daq.reqparameter('last_day', central=self.info['last_day'], fixed=True,
                          label='Last DAQ day end:    %s'%self.info['str_last_day'] )
        daq.reqparameter('ndays', central=self.info['days'], fixed=True,
                          label='Total number of days' )

        self.reqparameter('seconds_in_day', None, central=24.*60.*60., fixed=True, label='Number of seconds in day')
        self.reqparameter('days_in_second', None, central=1.0/(24.*60.*60.), fixed=True, label='Number of days in a second')

        data_lt = 0.0
        for it in self.nidx:
            ad, = it.current_values()
            data = self.data.get(ad, None)
            if data is None:
                raise self.exception('Failed to retrieve data for %s from %s'%(ad, self.cfg.file))

            data_lt = data['livetime']+data_lt
            livetime    = C.Points(data['livetime'], labels=it.current_format('Livetime\n{autoindex}'))
            eff         = C.Points(data['eff'],      labels=it.current_format('Efficiency (mu*mult)\n{autoindex}'))
            efflivetime = C.Product(livetime, eff,   labels=it.current_format('Livetime (eff)\n{autoindex}'))

            self.context.objects[('livetime',ad)] = livetime
            self.context.objects[('eff',ad)] = eff
            self.context.objects[('efflivetime',ad)] = efflivetime

            self.set_output('eff_daily',           it, eff.single())
            self.set_output('livetime_daily',      it, livetime.single())
            self.set_output('efflivetime_daily',   it, efflivetime.single())

        ndays_daq = (data_lt>0.0).sum()
        daq.reqparameter('ndays_daq', central=ndays_daq, fixed=True, sigma=0.01,
                          label='Total number of DAQ days (exclude no DAQ)' )


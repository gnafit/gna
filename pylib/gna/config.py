#!/usr/bin/env python

"""GNA configuration library
the global configuration is read from './config' folder
optional configuration overriding the global one may be stored in the './config_local' folder"""

from gna.configurator import configurator
cfg = configurator( '{location}/gna/gnacfg.py', subst='default', debug=False,
                    prefetch = False )

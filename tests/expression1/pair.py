#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
from gna.expression.preparse import *

s = 'echo ( alsdkfjlskjdf ( lsjdflksjdf  ) )'
print(s)
print(open_fcn(s))

s = 'ec|ho ( alsdkfjlskjdf ( lsjdflksjdf  ) )'
print(s)
print(open_fcn(s))

s = 'echo ( alsdk|fjlskjdf ( lsjdflksjdf  ) )'
print(s)
print(open_fcn(s))

s = 'echo ( alsdkfjlskjdf ( lsjd|flksjdf  ) )'
print(s)
print(open_fcn(s))

s = 'e|cho ( al|sdkfjlskjdf ( lsjd|flksjdf  ) )'
print(s)
print(open_fcn(s))

s = 'echo ( alsdkfjlskjdf |( lsjdflksjdf  ) )'
print(s)
print(open_fcn(s))

s = 'echo ( alsdkfjlskjdf (| lsjdflksjdf  ) )'
print(s)
print(open_fcn(s))


s = 'echo ( alsdkfjlskjdf ( lsjdflksjdf  |) )'
print(s)
print(open_fcn(s))

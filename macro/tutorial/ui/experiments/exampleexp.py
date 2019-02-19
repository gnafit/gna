# -*- coding: utf-8 -*-

from __future__ import print_function
from gna.exp import baseexp
import ROOT as R

class exp(baseexp):
    @classmethod
    def initparser(cls, parser, namespace):
        pass

    def __init__(self, namespace, opts):
        baseexp.__init__(self, namespace, opts)

        self.build()
        self.register()

    def build(self):
        import IPython; IPython.embed()

    def register(self):
        pass


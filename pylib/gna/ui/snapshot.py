"""Given an observable make it's snapshot via Snapshot transformation"""

from gna.ui import basecmd
import argparse
import os.path
from pkgutil import iter_modules
from gna.config import cfg
from gna import constructors as C
import sys

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('name_in', help='input observable name')
        parser.add_argument('name_out', help='observable name (output)')
        parser.add_argument('--ns', help='namespace')
        parser.add_argument('-H', '--hidden', action='store_true', help='make output hidden')
        parser.add_argument('-l', '--label', help='Snapshot node label')

    def __init__(self, *args, **kwargs):
        basecmd.__init__(self, *args, **kwargs)

    def init(self):
        self.ns = self.env.globalns(self.opts.ns)
        try:
            output = self.ns.getobservable(self.opts.name_in)
        except KeyError:
            output = self.env.future['spectra', self.opts.name_in]

        if not output:
            raise Exception('Invalid or missing output: {}'.format(self.opts.name_in))

        self.snapshot = C.Snapshot(output)
        trans = self.snapshot.snapshot
        if self.opts.label:
            trans.setLabel(self.opts.label)
        trans.touch()
        self.ns.addobservable(self.opts.name_out, self.snapshot.single(), export=not self.opts.hidden)
        self.env.future['spectra', self.opts.name_out] = self.snapshot.single()

        self.env.parts.snapshot[self.opts.name_out] = self.snapshot

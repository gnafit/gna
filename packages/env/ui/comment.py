# encoding: utf-8

"""Commenting UI. Arguments are ignored."""

from __future__ import print_function
from gna.ui import basecmd
from argparse import REMAINDER

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('args', nargs=REMAINDER, help="ignored arguments. First argument should not start with '-'")

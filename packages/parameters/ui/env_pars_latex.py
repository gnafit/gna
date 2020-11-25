"""Recursively prints parameters as a latex table."""

from gna.ui import basecmd
from tools.dictwrapper import DictWrapper, DictWrapperVisitor
from collections import OrderedDict
from load import ROOT as R
from gna.parameters import DiscreteParameter
from tabulate import tabulate

class DictWrapperParsPrinter(DictWrapperVisitor):
    _header = [ 'Key', 'Central', 'Sigma', 'Sigma, %', 'Comments', 'Label']
    def __init__(self, title):
        self._title = title
        self._data = []

        self._readers = {
                R.GaussianParameter('double'):     self._read_gaussian,
                R.UniformAngleParameter('double'): self._read_angle,
                DiscreteParameter:                 self._read_discrete,
                }

    def start(self, d):
        pass

    def stop(self, d):
        pass

    def write(self, outputs):
        options = dict()
        for out in outputs:
            if out.endswith('.tex'):
                t = tabulate(self._data, self._header, tablefmt='latex_booktabs', **options)
            else:
                t = tabulate(self._data, self._header, **options)

            if out=='-':
                print(t)
            else:
                with open(out, 'w') as f:
                    f.write(t)
                print('Write output file:', out)

    def enterdict(self, k, d):
        # if not k:
            # return
        # key = '.'.join(k)
        # self._data.append([key])
        pass

    def exitdict(self, k, d):
        pass

    def visit(self, k, v):
        depth = len(k)
        key = '.'.join(k)
        self._read_par(key, v)

    def _read_par(self, key, par):
        entry=self._readers.get(type(par), self._read_other)(key, par)

        if entry is not None:
            self._data.append(entry.values())

    def _read_gaussian(self, key, par):
        entry = OrderedDict(key=key)

        free   = par.isFree()
        fixed  = par.isFixed()
        corr   = par.isCorrelated()
        biased = par.value()!=par.central()

        entry['value']=par.central()
        if par.isFixed():
            entry['unc'] = ''
            entry['relunc'] = ''
        elif free:
            entry['unc'] = 'free'
            entry['relunc'] = ''
        else:
            entry['unc'] = par.sigma()
            if par.central():
                entry['relunc'] = par.sigma()/par.central()*100.0
            else:
                entry['relunc'] = ''

        marks = []
        if corr: marks.append('correlated')
        if biased: marks.append('modified')
        entry['marks'] = ', '.join(marks)
        entry['label'] = par.label()

        return entry

    def _read_discrete(self, key, par):
        entry = OrderedDict(key=key)
        variants = par.getVariants()
        biased = par.default!=par.value() if par.default else False

        entry['value']=par.value()
        entry['unc'] = 'discrete'
        entry['relunc'] = ''

        marks = [ '[{}]'.format(', '.join(variants)) ]
        if biased: marks.append('modified')

        entry['marks'] = ', '.join(marks)
        entry['label'] = par.label()

        return entry

    def _read_angle(self, key, par):
        entry = OrderedDict(key=key)

        free   = par.isFree()
        fixed  = par.isFixed()
        biased = par.value()!=par.central()

        entry['value']=r'{}'.format(par.central())
        if par.isFixed():
            entry['unc'] = ''
            entry['relunc'] = ''
        else:
            entry['unc'] = 'free'
            entry['relunc'] = ''

        marks = []
        if biased: marks.append('modified')
        entry['marks'] = ', '.join(marks)
        entry['label'] = par.label()

        return entry

    def _read_other(self, key, par):
        pass

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('paths', nargs='*', default=((),), help='paths to print')
        parser.add_argument('-o', '--output', nargs='+', default=['-'], help='latex file to write, `-` for stdout')

    def init(self):
        self.build_dict()
        printer = DictWrapperParsPrinter('')
        self.storage.visit(printer)
        printer.write(self.opts.output)

    def build_dict(self):
        self.storage = DictWrapper(OrderedDict(), split='.')
        for path in self.opts.paths:
            ns = self.env.globalns(path)
            for name, par in ns.walknames():
                self.storage[name]=par

    __tldr__ = """\
                The module enables the user to create a latex table for parameters.
                It accepts multiple paths with `env` (not `env.future`) and prints a text table to the stdout
                and a latex table to the file, provided after an '-o' option.

                \033[32mPrint the parameters to the file 'output.tex':
                ```sh
                ./gna \\
                    -- gaussianpeak --name peak \\
                    -- env-pars-latex peak -o output.tex
                ```

                The module uses python module [tabulate](https://github.com/astanin/python-tabulate) for printing.
               """

"""Recursively prints parameters as a latex table."""

from gna.ui import basecmd
from tools.dictwrapper import DictWrapper, DictWrapperVisitor
from load import ROOT as R
from gna.parameters import DiscreteParameter
from tabulate import tabulate
from env.lib.cwd import update_namespace_cwd
from typing import List
import re

_labels = {
        'unc':    'Sigma',
        'relunc': 'Sigma, %',
        'comments':  'Comments',
        }

_mask_enumerated_start = re.compile('^.*_0+[01]$')
_mask_enumerated = re.compile(r'^(.*)_\d{2,}$')

class DictWrapperParsPrinter(DictWrapperVisitor):
    _columns: List[str] = [ 'number', 'key', 'value', 'unc', 'relunc', 'comments', 'label' ]
    _post_processed: bool = False

    _similar_compress_after: int = 0
    _similar_compress_current_group: bool = False
    _similar_match: str = None

    _counter: int = 0
    _counter_local: int = 0
    def __init__(self, title, groups=True, skip_types=(), fmt=None, columns=None, compress_similar_after=0):
        self._title = title
        self._raw_data = []
        self._data = []
        self._fmt = fmt

        self._similar_compress_after = compress_similar_after
        self._groups = groups
        if columns:
            self._columns = columns
        elif groups:
            self._columns[1] = 'name'

        self._table = dict((col, _labels.get(col, col.capitalize())) for col in self._columns)

        self._skip_evaluated   = 'evaluated'   in skip_types
        self._skip_fixed       = 'fixed'       in skip_types
        self._skip_variable    = 'variable'    in skip_types
        self._skip_free        = 'free'        in skip_types
        self._skip_constrained = 'constrained' in skip_types
        self._skip_other       = 'other'       in skip_types
        self._skip_discrete    = 'discrete'    in skip_types

        self._readers = {
                R.GaussianParameter('double'):     self._read_gaussian,
                R.UniformAngleParameter('double'): self._read_angle,
                R.Variable('double'):              self._read_variable,
                DiscreteParameter:                 self._read_discrete,
                }

    def start(self, d):
        self._counter=0

    def stop(self, d):
        lastgroup = None
        skip=0

        def printskip():
            if skip:
                self._data.append(['', f'Skip {skip} similar lines'])

        for entry in self._raw_data:
            if not entry['similar_number']:
                printskip()
                skip=0

            if self._similar_compress_after and entry['similar_number']>=self._similar_compress_after:
                skip+=1
                continue

            if self._groups:
                newgroup = entry['group']
                if lastgroup is None or lastgroup!=newgroup:
                    lastgroup = newgroup
                    self._data.append(['', newgroup])

            line = [entry.get(k) for k in self._table]
            self._data.append(line)
        else:
            printskip()

        self._post_processed = True

    def write(self, outputs):
        options = dict()
        for out in outputs:
            if out.endswith('.tex'):
                t = tabulate(self._data, self._table.values(), tablefmt='latex_booktabs', **options)
            else:
                t = tabulate(self._data, self._table.values(), **options)

            if out:
                with open(out, 'w') as f:
                    f.write(t)
                print('Write output file:', out)
            else:
                print(t)

    def enterdict(self, k, d):
        self._counter_local=0
        self._similar_compress_current_group=False

    def exitdict(self, k, d):
        pass

    def visit(self, k, v):
        assert not self._post_processed
        group_current = '.'.join(k[:-1])
        name_current = k[-1]

        key = '.'.join(k)
        depth = len(k)

        entry = {
                'key': key,
                'depth': depth,
                'group': group_current,
                'name': name_current,
                'value': '',
                'unc': '',
                'relunc': '',
                'similar_number': 0
                }

        # Read the data
        entry = self._read_par(key, v, entry)
        if not entry:
            return

        self._determine_similar_number(entry)

    def _determine_similar_number(self, entry):
        name_current = entry['name']
        #
        # Check for similarities
        #
        if not self._similar_compress_after:
            return

        if self._counter_local: # check if the first element has name similar to 'something_00'
            return

        if _mask_enumerated_start.match(name_current):
            self._similar_compress_current_group=True

        if not self._similar_compress_current_group:
            return

        match = _mask_enumerated.match(name_current)
        if not match:
            return

        similar_match = match.group(1)

        try:
            newhash = entry['hash'] = f'{similar_match}={entry["value"]:.8e}; s={entry["unc"]:.8e}; sr={entry["relunc"]:.8e}'
        except ValueError:
            newhash = entry['hash'] = f'{similar_match}={entry["value"]}; s={entry["unc"]}; sr={entry["relunc"]}'

        try:
            preventry = self._raw_data[-2]
        except IndexError:
            return

        if newhash==preventry.get('hash', ''):
            entry['similar_number']=preventry['similar_number']+1

    def _read_par(self, key, par, entry):
        entry=self._readers.get(type(par), self._read_other)(key, par, entry)

        if entry is None:
            return

        self._counter+=1
        entry['number']  = self._counter # 1-based


        if self._fmt:
            for k in ('value', 'unc', 'relunc'):
                val = entry[k]
                if isinstance(val, float):
                    entry[k]=self._fmt.format(val)

        self._raw_data.append(entry)
        return entry

    def _read_variable(self, key, par, entry):
        if self._skip_evaluated:
            return

        size = par.getVariable().size()
        comments=['evaluated']
        value = par.value()
        if size>1:
            if size==2:
                value+=1j*par.getVariable().value(1)
            comments+=[f'n={size}']
        entry.update({
                'comments': ', '.join(comments),
                'label': par.label(),
                })

        return entry

    def _read_gaussian(self, key, par, entry):
        free     = par.isFree()
        fixed    = par.isFixed()
        variable = not fixed
        constrained = variable and not free

        if free  and self._skip_free: return
        if fixed and self._skip_fixed: return
        if variable and self._skip_variable: return
        if constrained and self._skip_constrained: return

        corr   = par.isCorrelated()
        biased = par.value()!=par.central()

        entry['value']=par.value()
        if par.isFixed():
            pass
        elif free:
            entry['unc'] = 'free'
        else:
            entry['unc'] = par.sigma()
            if par.central():
                entry['relunc'] = par.sigma()/par.central()*100.0

        comments = []
        if corr: comments.append('correlated')
        if biased: comments.append('modified')
        entry['comments'] = ', '.join(comments)
        entry['label'] = par.label()

        return entry

    def _read_discrete(self, key, par, entry):
        if self._skip_discrete: return

        variants = par.getVariants()
        biased = par.default!=par.value() if par.default else False

        entry['value']=par.value()
        entry['unc'] = 'discrete'

        comments = [ '[{}]'.format(', '.join(variants)) ]
        if biased: comments.append('modified')

        entry['comments'] = ', '.join(comments)
        entry['label'] = par.label()

        return entry

    def _read_angle(self, key, par, entry):
        free   = par.isFree()
        fixed  = par.isFixed()
        variable = not fixed

        if free  and self._skip_free: return
        if fixed and self._skip_fixed: return
        if variable and self._skip_variable: return

        biased = par.value()!=par.central()

        entry['value']=r'{}'.format(par.central())
        if not par.isFixed():
            entry['unc'] = 'free'

        comments = []
        if biased: comments.append('modified')
        entry['comments'] = ', '.join(comments)
        entry['label'] = par.label()

        return entry

    def _read_other(self, key, par, entry):
        if self._skip_other:
            return

        entry['comments']='failed to print'
        return netry

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('paths', nargs='*', default=((),), help='paths to print')
        parser.add_argument('-o', '--output', nargs='+', default=[], help='latex file to write, `-` for stdout')
        parser.add_argument('-s', '--stdout', action='store_true', help='print to stdout')
        parser.add_argument('--skip-types', nargs='+', default=[], choices=('evaluated', 'fixed', 'variable', 'free', 'constrained', 'other', 'discrete'), help='types to skip')
        parser.add_argument('--fmt', help='number format')
        parser.add_argument('--columns', nargs='+', help='columns')
        parser.add_argument('--compress-after', type=int, default=0, help='compress similar items after N occurances', metavar='N')

    def init(self):
        update_namespace_cwd(self.opts, 'output')

        self.build_dict()
        printer = DictWrapperParsPrinter('', skip_types=self.opts.skip_types,
                                         fmt=self.opts.fmt, columns=self.opts.columns,
                                         compress_similar_after=self.opts.compress_after)
        self.storage.visit(printer)

        writepath = self.opts.output
        if self.opts.stdout:
            writepath+=['']
        printer.write(writepath)

    def build_dict(self):
        self.storage = DictWrapper(dict(), split='.')
        for path in self.opts.paths:
            try:
                par=self.env.globalns[path]
            except KeyError:
                ns = self.env.globalns(path)
                for name, par in ns.walknames():
                    self.storage[name]=par
            else:
                self.storage[path]=par


    __tldr__ = """\
                The module enables the user to create a latex table for parameters.
                It accepts multiple paths with `env` (not `env.future`) and prints a text table to the stdout
                and a latex table to the file, provided after an `-o` option.

                Print the parameters to the file 'output.tex':
                ```sh
                ./gna \\
                    -- gaussianpeak --name peak \\
                    -- env-pars-latex peak -o output.tex
                ```

                The module uses python module [tabulate](https://github.com/astanin/python-tabulate) for printing.
               """

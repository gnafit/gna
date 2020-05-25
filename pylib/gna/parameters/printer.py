# -*- coding: utf-8 -*-

from __future__ import print_function
from __future__ import absolute_import
from load import ROOT
import numpy as N
from gna.parameters import DiscreteParameter
from gna.bindings import provided_precisions

from itertools import chain, tee, cycle
import itertools

from gna.bindings import patchROOTClass

try:
    import sys
    if not sys.stdout.isatty():
        raise RuntimeError()
    from colorama import Fore, Style

    def colorize(string, color):
        return color + str(string) + Style.RESET_ALL

except:
    def colorize(string, color):
        return string

    class DummyAttr(object):
        def __getattr__(self, attr):
            return ''
    Style = DummyAttr()
    Fore = DummyAttr()

unctypes = ()
for precision in provided_precisions:
    unctypes += ( ROOT.Variable(precision), ROOT.Variable('complex<%s>'%precision) )
unctypes+=(DiscreteParameter,)

namefmt='{name:30}'

valfmt='={color}{val:11.6g}'
valdfmt='={color}{val:>11s}'

centralfmt='{central:11.6g}'

cvalfmt='={color}{rval:11.6g}+i{ival:11.6g}'

ccentralfmt='{rcentral:11.6g}+i{icentral:11.6g}'

sigmafmt='± {sigma:11.6g}'
limitsfmt=' ({:11.6g}, {:11.6g})'
centralsigmafmt= centralfmt + sigmafmt
relsigmafmt=' [{relsigma:11.6g}%]'
npifmt     =' [{npi:11.6g}π]'

centralsigma_len=len(centralsigmafmt.format(central=0, sigma=0))
central_len=len(centralfmt.format(central=0))
sigma_len=len(sigmafmt.format(sigma=0))
relsigma_len=len(relsigmafmt.format(relsigma=0))

central_reg_len = centralsigma_len+relsigma_len-1
centralrel_empty =(central_reg_len)*' '
sigmarel_empty =(sigma_len+relsigma_len-1)*' '
sigma_empty =(sigma_len-1)*' '

sepstr='{} │ '.format(Style.RESET_ALL)

fixedstr='[fixed]'
fixedstr_len = (centralsigma_len+relsigma_len-1-len(fixedstr))
fixed_half_width = int(fixedstr_len/2)
fixedstr = fixed_half_width*' ' + fixedstr + fixed_half_width*" "

freestr =' [free]'

freestr+=' '*(relsigma_len-len(freestr))

variants_fmt = ' {variants:^{width}s}'

class CovarianceStore():
    def __init__(self):
        self.storage = list()

    def add_to_store(self, par):
        if not any(self.__in_store(par)):
            self.storage.append(list(chain((par,), par.getAllCorrelatedWith())))
        else:
            return

    def __len__(self):
        return len(self.storage)


    def __in_store(self, par):
        for par_set in self.storage:
            for item in par_set:
                yield par == item

def print_correlated_parameters_block(par_set, correlations='short'):
    max_offset = max((len(x.qualifiedName()) for x in par_set))

    npars = len(par_set)
    if correlations=='short' and npars>10:
        print('{} parameters: first is {}'.format(npars, par_set[0].qualifiedName()))
    else:
        for pivot in par_set:
            full_name = pivot.qualifiedName()
            s = colorize(full_name, Fore.CYAN)
            current_offset = len(full_name)
            if max_offset != current_offset:
                initial_sep = " "*(max_offset-current_offset +1)
            else:
                initial_sep = " "
            s += initial_sep
            for par in par_set:
                s += '{:6g}'.format(par.getCorrelation(pivot))
                s += " "
            print(s)

def print_correlated_parameters(cor_store, correlations='short'):
    if len(cor_store) == 0:
        return
    raw = "\nCorrelations between parameters:"
    title = colorize(raw, Fore.RED)
    print(title)
    for par_set in cor_store.storage:
        print_correlated_parameters_block(par_set, correlations)
        print("")

def print_parameters(ns, recursive=True, labels=False, cor_storage=None, stats=None, correlations='short'):
    '''Pretty prints parameters in a given namespace. Prints parameters
    and then outputs covariance matrices for correlated pars. '''
    if cor_storage is None:
        cor_storage = CovarianceStore()
        top_level = True
    else:
        top_level = False

    header = False
    for name, var in ns.items():
        if isinstance( ns.storage[name], str ):
            print(u'  {name:30}-> {target}'.format( name=name, target=ns.storage[name] ))
            continue
        if not isinstance( var, unctypes ):
            continue
        if not header:
            print("Variables in namespace '{}':".format(colorize(ns.path, color=Fore.GREEN)))
            header=True

        try:
            if var.isCorrelated():
                cor_storage.add_to_store(var)
        except (AttributeError, TypeError):
            pass

        print(end='  ')
        print(var.__str__(labels=labels))
        varstats(var, stats)
    if recursive:
        for sns in ns.namespaces.values():
            print_parameters(sns, recursive=recursive, labels=labels, cor_storage=cor_storage, stats=stats)

    if correlations and top_level:
        print_correlated_parameters(cor_storage, correlations)

def varstats(var, stats):
    if stats is None:
        return

    def increment(name):
        stats[name]=stats.get(name, 0)+1

    if stats is not None:
        tv=type(var).__name__
        increment('total')
        if tv.startswith('GaussianParameter') or tv.startswith('UniformAngleParameter'):
            if var.isFixed():
                increment('fixed')
            else:
                increment('variable')
                if var.isFree():
                    increment('free')
                else:
                    increment('constrained')
        elif tv.startswith('Variable'):
            increment('evaluable')
        else:
            increment('unknown')

@patchROOTClass( [ROOT.Variable(prec) for prec in provided_precisions], '__str__' )
def Variable__str( self, labels=False ):
    var = self.getVariable()
    size = var.size()
    if size==2:
        return Variable_complex__str(self, labels, value=var.values().complex())

    name = self.name()
    if size>2:
        name += ' [%i]'%size

    fmt = dict(
            name    = colorize(self.name(), Fore.CYAN),
            val     = self.value(),
            color = Fore.BLUE
            )
    label = self.label()
    if not labels or label=='value':
        label=''

    s= namefmt.format(**fmt)
    s+=valfmt.format(**fmt)

    if labels:
        s+=sepstr+centralrel_empty+sepstr
    if label:
        s+= Fore.LIGHTGREEN_EX + label + Style.RESET_ALL

    s += Style.RESET_ALL
    return s

@patchROOTClass( [ROOT.Variable('complex<%s>'%prec) for prec in provided_precisions], '__str__' )
def Variable_complex__str(self, labels=False, value=None):
    if value is None:
        value = self.values()
    fmt = dict(
            name  = colorize(self.name(), Fore.CYAN),
            rval  = value.real(),
            ival  = value.imag(),
            color = Fore.BLUE
            )
    label = self.label()
    if not labels or label=='value':
        label=''

    s= namefmt.format(**fmt)
    s+=cvalfmt.format(**fmt)

    # cnum = value.real() + value.imag()*1j
    # angle = N.angle(cnum, deg=True)
    # mag = N.absolute(cnum)

    if labels:
         s+=sepstr+centralrel_empty[:-central_len-2]+sepstr
    if label:
        s+= Fore.LIGHTGREEN_EX + label + Style.RESET_ALL

    s += Style.RESET_ALL
    return s

@patchROOTClass( [ROOT.UniformAngleParameter(prec) for prec in provided_precisions], '__str__' )
def UniformAngleParameter__str( self, labels=False  ):
    fmt = dict(
            name    = colorize(self.name(), Fore.CYAN),
            val     = self.value(),
            central = self.central(),
            npi     = self.value()/N.pi,
            color   = Fore.BLUE
            )
    label = self.label()
    if not labels or label=='value':
        label=''

    s= namefmt.format(**fmt)
    s+=valfmt.format(**fmt)

    if self.isFixed():
        s+=sepstr+fixedstr
    else:
        s+= sepstr+centralfmt.format(**fmt)
        s+= sigma_empty
        s+= npifmt.format(**fmt)

        s+=sepstr+' (-π, π)                   '

    if labels:
        s+=sepstr
    if label:
        s+= Fore.LIGHTGREEN_EX + label + Style.RESET_ALL

    s += Style.RESET_ALL
    return s

@patchROOTClass( [ROOT.GaussianParameter(prec) for prec in provided_precisions], '__str__' )
def GaussianParameter__str( self, labels=False  ):
    fmt = dict(
            name    = colorize(self.name(), Fore.CYAN),
            val     = self.value(),
            central = self.central(),
            sigma   = self.sigma(),
            color   = Fore.BLUE
            )
    correlated = self.isCorrelated()
    limits  = self.limits()
    label = self.label()
    if not labels or label=='value':
        label=''

    s= namefmt.format(**fmt)
    s+=valfmt.format(**fmt)

    if self.isFixed():
        s += sepstr
        s += Fore.LIGHTYELLOW_EX + fixedstr
    else:
        s+=sepstr + centralsigmafmt.format(**fmt)
        if self.isFree():
            s += Fore.LIGHTYELLOW_EX + freestr
        else:
            if fmt['central']:
                s+=relsigmafmt.format(relsigma=fmt['sigma']/fmt['central']*100.0)
            else:
                s+=' '*relsigma_len

        if correlated:
            s += Fore.LIGHTGREEN_EX
            s += " [C]"

        if limits.size():
            s+=sepstr
            for (a, b) in limits:
                s+=limitsfmt.format(a, b)

    if labels:
        s+=sepstr

    if label:
        s+= Fore.LIGHTGREEN_EX + label + Style.RESET_ALL

    s += Style.RESET_ALL

    return s

def DiscreteParameter____str__(self, labels=False):
    fmt = dict(
        name    = colorize(self.name(), Fore.CYAN),
        val     = self.value(),
        variants = str(self.getVariants()),
        color   = Fore.BLUE
        )
    label = self.getLabel()

    s= namefmt.format(**fmt)
    s+=valdfmt.format(**fmt)
    s+=sepstr+variants_fmt.format(width=central_reg_len-1, **fmt)

    if labels:
        s+=sepstr

    if label:
        s+= Fore.LIGHTGREEN_EX + label + Style.RESET_ALL

    return s
DiscreteParameter.__str__ = DiscreteParameter____str__

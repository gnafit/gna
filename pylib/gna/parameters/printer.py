# -*- coding: utf-8 -*-

from __future__ import print_function
from load import ROOT
import numpy as N
from gna.parameters import DiscreteParameter
from gna.bindings import provided_precisions

from itertools import chain, tee, cycle
import itertools

from gna.bindings import patchROOTClass
try:
    from colorama import Fore, Style
    colorama_present = True
except ImportError:
    colorama_present = False

def colorize(string, color):
    return color + str(string) + Style.RESET_ALL

unctypes = ( ROOT.Variable('double'), ROOT.Variable('complex<double>'), DiscreteParameter )

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

if colorama_present:
    sepstr='{} │ '.format(Style.RESET_ALL)
else:
    sepstr=' │ '

fixedstr='[fixed]'
fixedstr_len = (centralsigma_len+relsigma_len-1-len(fixedstr))
fixedstr = (fixedstr_len/2)*' ' + fixedstr + (fixedstr_len/2)*" "

freestr =' [free]'

freestr+=' '*(relsigma_len-len(freestr))

variants_fmt = ' {variants:^{width}s}'

class CovarianceStore():
    def __init__(self):
        self.storage = list()

    def add_to_store(self, par):
        if not any(self.__in_store(par)):
            self.storage.append(set(chain((par,), par.getAllCovariatedWith())))
        else:
            return

    def __len__(self):
        return len(self.storage)


    def __in_store(self, par):
        for par_set in self.storage:
            for item in par_set:
                yield par == item

def print_covariated(cov_store):
    if len(cov_store) == 0:
        return
    raw = "\nCorrelations between parameters:"
    title = colorize(raw, Fore.RED) if colorama_present else raw
    print(title)
    for par_set in cov_store.storage:
        max_offset = max((len(x.qualifiedName()) for x in par_set))
        for pivot in par_set:
            full_name = pivot.qualifiedName()
            s = colorize(full_name, Fore.CYAN) if colorama_present else full_name
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
        print("")


def print_parameters( ns, recursive=True, labels=False, cov_storage=None, stats=None):
    '''Pretty prints parameters in a given namespace. Prints parameters
    and then outputs covariance matrices for covariated pars. '''
    if cov_storage is None:
        cov_storage = CovarianceStore()
        top_level = True
    else:
        top_level = False

    header = False
    for name, var in ns.iteritems():
        if isinstance( ns.storage[name], str ):
            print(u'  {name:30}-> {target}'.format( name=name, target=ns.storage[name] ))
            continue
        if not isinstance( var, unctypes ):
            continue
        if not header:
            if colorama_present:
                print("Variables in namespace '{}':".format(colorize(ns.path, color=Fore.GREEN)))
            else:
                print("Variables in namespace '%s'"%ns.path)
            header=True

        try:
            if var.isCovariated():
                cov_storage.add_to_store(var)
        except (AttributeError, TypeError):
            pass

        print(end='  ')
        print(var.__str__(labels=labels))
        varstats(var, stats)
    if recursive:
        for sns in ns.namespaces.itervalues():
            print_parameters(sns, recursive=recursive, labels=labels, cov_storage=cov_storage, stats=stats)

    if top_level:
        print_covariated(cov_storage)

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
            name    = colorize(name, Fore.CYAN) if colorama_present else name,
            val     = self.value(),
            color = Fore.BLUE if colorama_present else ""
            )
    label = self.label()
    if not labels or label=='value':
        label=''

    s= namefmt.format(**fmt)
    s+=valfmt.format(**fmt)

    if labels:
        s+=sepstr+centralrel_empty+sepstr
    if label:
        s+= Fore.LIGHTGREEN_EX + label + Style.RESET_ALL if colorama_present else label

    s += Style.RESET_ALL if colorama_present else ""
    return s

@patchROOTClass( [ROOT.Variable('complex<%s>'%prec) for prec in provided_precisions], '__str__' )
def Variable_complex__str(self, labels=False, value=None):
    if value is None:
        value = self.value()
    fmt = dict(
            name  = colorize(self.name(), Fore.CYAN) if colorama_present else self.name(),
            rval  = value.real(),
            ival  = value.imag(),
            color = Fore.BLUE if colorama_present else ""
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
        s+= Fore.LIGHTGREEN_EX + label + Style.RESET_ALL if colorama_present else label

    s += Style.RESET_ALL if colorama_present else ""
    return s

@patchROOTClass( [ROOT.UniformAngleParameter(prec) for prec in provided_precisions], '__str__' )
def UniformAngleParameter__str( self, labels=False  ):
    fmt = dict(
            name    = colorize(self.name(), Fore.CYAN) if colorama_present else self.name(),
            val     = self.value(),
            central = self.central(),
            npi     = self.value()/N.pi,
            color   = Fore.BLUE if colorama_present else ""
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
        s+= Fore.LIGHTGREEN_EX + label + Style.RESET_ALL if colorama_present else label

    s += Style.RESET_ALL if colorama_present else ""
    return s

@patchROOTClass( [ROOT.GaussianParameter(prec) for prec in provided_precisions], '__str__' )
def GaussianParameter__str( self, labels=False  ):
    fmt = dict(
            name    = colorize(self.name(), Fore.CYAN) if colorama_present else self.name(),
            val     = self.value(),
            central = self.central(),
            sigma   = self.sigma(),
            color   = Fore.BLUE if colorama_present else ""
            )
    covariated = self.isCovariated()
    limits  = self.limits()
    label = self.label()
    if not labels or label=='value':
        label=''

    s= namefmt.format(**fmt)
    s+=valfmt.format(**fmt)

    if self.isFixed():
        s += sepstr
        s += Fore.LIGHTYELLOW_EX + fixedstr if colorama_present else fixedstr
    else:
        s+=sepstr + centralsigmafmt.format(**fmt)
        if self.isFree():
            s += Fore.LIGHTYELLOW_EX + freestr if colorama_present else freestr
        else:
            if fmt['central']:
                s+=relsigmafmt.format(relsigma=fmt['sigma']/fmt['central']*100.0)
            else:
                s+=' '*relsigma_len

        if covariated:
            s += sepstr
            s += Fore.LIGHTGREEN_EX if colorama_present else ""
            s += "[C]"

        if limits.size():
            s+=sepstr
            for (a,b) in limits:
                s+=limitsfmt.format(a,b)

    if labels:
        s+=sepstr

    if label:
        s+= Fore.LIGHTGREEN_EX + label + Style.RESET_ALL if colorama_present else label

    s += Style.RESET_ALL if colorama_present else ""

    return s

def DiscreteParameter____str__(self, labels=False):
    fmt = dict(
        name    = colorize(self.name(), Fore.CYAN) if colorama_present else self.name(),
        val     = self.value(),
        variants = str(self.getVariants()),
        color   = Fore.BLUE if colorama_present else ""
        )
    label = self.getLabel()

    s= namefmt.format(**fmt)
    s+=valdfmt.format(**fmt)
    s+=sepstr+variants_fmt.format(width=central_reg_len-1, **fmt)

    if labels:
        s+=sepstr

    if label:
        s+= Fore.LIGHTGREEN_EX + label + Style.RESET_ALL if colorama_present else label

    return s
DiscreteParameter.__str__ = DiscreteParameter____str__


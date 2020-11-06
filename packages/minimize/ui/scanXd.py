#!/usr/bin/env python

"""Implements simple N-dimensional scanning with optional minimization for each point"""

from gna.ui import basecmd, set_typed, append_typed
from tools.argparse_utils import AppendSubparser
import ROOT
from gna.parameters.parameter_loader import get_parameters
from collections import OrderedDict
import time
import numpy as np
import itertools as I
from tools.terminal import progress
from pprint import pprint

class scancmd(basecmd):
    localenv = None
    variables = None
    ndim = 0
    grid_function = None
    output_hdf = None

    @classmethod
    def initparser(cls, parser, env):
        #
        # Make variable parser
        #
        import argparse
        varparser = AppendSubparser( prefix_chars='+', prog='--var' )
        varparser.add_argument( 'parname', help='variable name' )
        varparser.add_argument( '+L', '++log', action='store_true', help='use logarithmic scale' )
        # varparser.add_argument( '++split', default=1, type=int, help='split the range into <N> subranges. To be used with --split key', metavar='N' )
        # varparser.add_argument( '+v', '++value', help='saved value to use for center' )

        group = varparser.add_mutually_exclusive_group(required=True)
        group.add_argument( '+l', '++limits', nargs=2, type=float, help='variable limits')
        group.add_argument( '+e', '++err', type=float, help='error to set limits (+/-e)')

        group = varparser.add_mutually_exclusive_group(required=True)
        group.add_argument( '+s', '++step', type=float, help='variable step' )
        group.add_argument( '+n', '++npoints', type=int, help='number of points' )

        #
        # Make main parser
        #
        parser.add_argument('minimizer', default=[], metavar=('MINIMIZER'), action=append_typed(env.parts.minimizer))
        parser.add_argument('--pars', nargs='+', required=True, help='All parameters track')
        # parser.add_argument( '--file', default='main', help='select target file' )
        # parser.add_argument( '--hdf', help='save hdf output' )

        parser.add_argument( '-v', '--var', nargs='+', action=varparser,
                             help='variables specification, use +h/++help for syntax', dest='variables' )
        parser.add_argument( '-f', '--folder', default='scanXd', help='folder to store the data' )
        parser.add_argument( '-V', '--verbose', action='count', default=0, help='verbose' )
        parser.add_argument( '-s', '--scan', action='store_true', help='do scan' )

        group = parser.add_mutually_exclusive_group()
        group.add_argument( '-m', '--minimize', action='store_true', help='minimize on each step' )
        group.add_argument( '--dummy', action='store_true', help='run script without actual calculation' )
        group.add_argument( '--dummyfalse', action='store_true', help='run script without actual calculation, with failure status' )

        parser.add_argument( '--start-from', choices=[ 'default', 'bestfit' ], default='default', help='the very starting point' )
        # parser.add_argument( '--minpoint', choices=[ 'initial', 'previous', 'initialonfail' ], default='initialonfail', help='the initial minimization configuration for netx point' )

        # parser.add_argument( '--split', type=int, help='index of the sub region to scan' )

    def init(self):
        nvars = len(self.opts.variables)
        assert nvars and (self.ndim is None or nvars==self.ndim), 'Can scan only %i variable(s)'%( self.ndim )

        self.minimizer = self.opts.minimizer[0]

        self.allpars  = OrderedDict([(par.qualifiedName(), par) for par in get_parameters(self.opts.pars, drop_fixed=True, drop_free=False, drop_constrained=False)])
        self.scanpars = OrderedDict([(par.qualifiedName(), par) for par in get_parameters([par.parname for par in self.opts.variables])])
        self.otherpars = [par for (name, par) in self.allpars.items() if not name in self.scanpars.keys()]

        # scanpars = []
        for varcfg in self.opts.variables:
            spar = self.allpars.get(varcfg.parname)
            # self.splits.append( varcfg.split )
        # self.splitmat = np.arange( np.prod( self.splits ) ).reshape( self.splits )

        # self.scanpars.Print()
        # if self.otherpars.GetNdim():
            # self.otherpars.Print()

        # if self.opts.hdf:
            # self.output_hdf = self.env.output_hdf[ self.opts.hdf ]

    def pushpars(self):
        for par in self.allpars.values():
            par.push()
    def poppars(self):
        for par in self.allpars.values():
            par.pop()
    def pushpars_scan(self):
        for par in self.scanpars.values():
            par.push()
    def poppars_scan(self):
        for par in self.scanpars.values():
            par.pop()

    def run(self):
        if self.opts.scan:
            self.do_scan()

    def do_scan( self, *args, **kwargs ):
        pass

    def fix_scanpars(self, fix=True, minimize=True):
        if not minimize:
            return
        for name in self.scanpars.keys():
            self.minimizer.fixpar(name, fixed=fix)

    def scan_get_par(self, parname, limits, **kwargs):
        step    = kwargs.pop('step'   , None)
        npoints = kwargs.pop('npoints', None)
        log     = kwargs.pop('log'    , False)
        err     = kwargs.pop('err'    , None)
        # split   = kwargs.pop('split'  , None)
        vkey    = kwargs.pop('value'  , None)
        assert kwargs=={}, 'Not empty kwargs: '+str(kwargs)

        try:
            par = self.scanpars[parname]
        except KeyError:
            assert par, 'There is no defined parameter '+parname

        # User limits from arguments or get limits from error
        df = None
        if limits:
            lmin, lmax = limits
        elif err:
            if vkey:
                assert False
            else:
                df = par.central()
            err = float(err)
            lmin = df - err
            lmax = df + err
        else:
            assert False, 'Insufficient arguments for parameter '+parname
        #print( '    limits', lmin, lmax )

        # Switch to logspace if needed
        if log:
            lmin = np.log10( lmin )
            lmax = np.log10( lmax )
            #print( '    switch to log' )
            #print( '    limits', lmin, lmax )

        # Define step and number of points
        if npoints:
            step = (lmax - lmin)/( npoints - 1 )
        else:
            assert step, 'Insufficient arguments for parameter '+parname
            npoints = int( ( lmax-lmin )/step + 0.5 ) + 1
        #print( parname )
        #print( '    step=', step )
        #print( '    npoints=', npoints )

        # Make an array with edges and an array with centers
        centers = np.linspace( lmin, lmax, num=npoints)
        edges   = np.linspace( lmin-step*0.5, lmax+step*0.5, num=npoints+1 )
        #print( '    centers', lmin, lmax, npoints)
        #print( '    centers', centers )
        #print( '    edges', edges )
        print( 'Initialize scan parameter %10.10s: %4i points'
               ' from %10.3g to %10.3g with step %10.3g'%( parname, len( centers ), lmin, lmax, step ) )
        if not df is None:
            print( '  center:', df )

        # Return to normal space if needed
        if log:
            centers = 10**centers
            edges   = 10**edges
            #print( '    switch to log' )
            #print( '    centers', centers )
            #print( '    edges', centers )

        if self.opts.verbose:
            print( '  bin centers:', centers )

        # def get_sub( i, split ):
            # nsplit = int(np.ceil(centers.size/(split)))
            # # print( split, centers.size, '->', nsplit )
            # i1 = i*nsplit
            # i2 = i1+nsplit

            # return centers[ i1:i2 ], edges[ i1:i2+1 ]

        # if not self.opts.split is None:
            # idx = np.unravel_index( self.opts.split, self.splitmat.shape )
            # pidx = idx[ self.varorder[parname] ]

            # centers, edges = get_sub( pidx, split )

            # if self.opts.verbose:
                # print( '  split bin centers:', centers )

        return par, centers, edges

    def init_grid_function(self, mode):
        self.result = {}
        if mode=='dummy':
            def gf():
                return 0.0, True
            self.grid_function = gf
        elif mode=='dummyfalse':
            def gf():
                return 0.0, False
            self.grid_function = gf
        elif mode=='min':
            def gf():
                res = self.result = self.minimizer.fit()
                return res.fun, res.success
            self.grid_function = gf
        else:
            assert False
            # rchi2fcn = self.localenv.chi2fcn
            # exp = self.localenv.exp
            # def fcn( ld ):
                # exp.update()
                # res = rchi2fcn.Eval()
                # return res, True
            # self.grid_function = fcn

class cmd(scancmd):
    ndim = None

    def do_scan( self, **kwargs ):
        minimize = kwargs.get('minimize', self.opts.minimize)
        dummy = kwargs.get('dummy', self.opts.dummy)

        folder   = kwargs.get('folder', self.opts.folder)
        self.init_grid_function(dummy and 'dummy' or minimize and 'min' or '')
        self.pushpars() # Initial parameters [save]

        shape, pars, centers, edges, idxs = [], [], [], [], []
        for varopts in self.opts.variables:
            parI, centersI, edgesI = self.scan_get_par(**varopts.__dict__)
            pars.append( parI )
            centers.append( centersI )
            edges.append( edgesI )
            n = len( centersI )
            shape.append( n )
            idxs.append( range( n ) )
        meshes = [ np.zeros( shape=shape, dtype='d' ) for par in pars ]
        array   = np.zeros( shape=shape, dtype='d' )
        array_s = np.zeros( shape=shape, dtype='i' )

        clock_start = time.perf_counter()
        idx_done, idx_total = 0, array.size
        print( 'Start scanXd for parameters (%i points):'%array.size, [par.qualifiedName() for par in pars] )
        status = True
        for idx in I.product( *idxs ):
            for par, mesh, values, iv in I.izip( pars, meshes, centers, idx ):
                val = values[iv]
                self.minimizer.fixpar(par.qualifiedName(), val, True)
                mesh[idx] = val

            self.pushpars_scan()

            array[idx], status=self.grid_function()
            array_s[idx]=status

            if self.opts.verbose:
                print( '%3i [%i]:'%(idx_done, status), [c[i] for c,i in zip(centers,idx)], '->', array[idx] )

            time_elapsed = time.perf_counter() - clock_start
            if self.opts.verbose>1:
                pprint(self.result.__dict__)

            self.poppars_scan()

            # if idx_done:
                # progress( 'Scanning', idx_done+1, idx_total, time_elapsed )

            idx_done+=1
        print( '' )

        self.poppars() # Initial parameters [load]
        self.fix_scanpars(False, minimize=minimize)

        # if self.output_hdf:
            # grp = self.output_hdf.create_group( folder )
            # grp.create_dataset( 'data',   data=array )
            # grp.create_dataset( 'status', data=array_s )
            # grp.create_dataset( 'pars', data=[ par.GetName() for par in pars ] )

            # for sgrpname, lsts in ( ( 'centers', centers ),
                                    # ( 'edges', edges ),
                                    # ( 'meshes', meshes ) ):
                # sgrp = grp.create_group( sgrpname )
                # for par, lst in I.izip( pars, lsts ):
                    # sgrp.create_dataset( par.GetName(), data=lst )

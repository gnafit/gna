import os
import numpy as np
import argparse
import itertools
from itertools import chain, repeat, islice, product
import h5py
import time
from collections import namedtuple

from gna.pointtree import PointTree
from gna.grids import PointSet
from gna.ui import basecmd, set_typed, append_typed

MinimizerProperties = namedtuple('MinimizerProperties',
                                 ['local', 'idxes', 'nres', 'names',
                                  'fitvars'])

class SegmentValidation(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        nseg, cur_seg = values
        if nseg < 0:
            raise ValueError('Passed negative number of segments')
        if cur_seg < 0:
            raise ValueError('Passed negative index for current segments')
        if cur_seg >= nseg:
            raise ValueError('Request for segment index larger then number of segments')
        setattr(namespace, self.dest, values)


class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        PointSet.addargs(parser, env)
        parser.add_argument('--pullminimizer', action=set_typed(env.parts.minimizer))
        group = parser.add_mutually_exclusive_group()
        group.add_argument('--samples', type=int, default=None)
        parser.add_argument('--output', type=os.path.abspath, required=True)
        group = parser.add_mutually_exclusive_group()
        group.add_argument('--fcscan')
        group.add_argument('--points')
        parser.add_argument('--tolerance', type=float, default=1e-2)
        parser.add_argument('--randomize', nargs='+', action='append', default=[])
        parser.add_argument('--minimizer', dest='minimizers', default=[],
                            metavar=('MINIMIZER'),
                            action=append_typed(env.parts.minimizer))
        parser.add_argument('--toymc', action=set_typed(env.parts.toymc))
        parser.add_argument('--toymc-type', choices=['static', 'grid'])
        parser.add_argument('-v', '--verbose', action='count', default=0)
        parser.add_argument('--segments', nargs=2, action=SegmentValidation, type=int,
                metavar=('NSEGMENTS', 'CURRENT_SEGMENT'),
                help='Split the grid over which scan is performed into '
                'NSEGMENTS segments and scan only CURRENT_SEGMENT now')

    def makerandomizer(self, randdesc):
        assert len(randdesc) >= 1
        if randdesc[0] == 'uniform':
            vmin, vmax = float(randdesc[1]), float(randdesc[2])
            return lambda random: random.uniform(vmin, vmax)
        elif randdesc[0] == 'normal':
            vmean, vsigma = float(randdesc[1]), float(randdesc[2])
            return lambda random: random.normal(vmin, vsigma)
        elif randdesc[0] == 'normal_limited':
            vmean, vsigma = float(randdesc[1]), float(randdesc[2])
            vmin, vmax = float(randdesc[3]), float(randdesc[4])
            def normal_limited(random):
                it = (random.normal(vmean, vsigma) for _ in itertools.count())
                itlim = itertools.dropwhile(lambda v: not vmin < v < vmax, it)
                return next(itlim)
            return normal_limited
        else:
            assert False

    def points(self):
        def randomseed():
            return np.random.randint(0, 4294967296)
        def getsampling(path):
            return randomseed(), self.opts.samples
        if self.opts.fcscan or self.opts.points:
            tree = PointTree(self.env, self.opts.fcscan or self.opts.points, "r")
            pointset = PointSet.fromtree(self.opts, tree)
            if self.opts.fcscan:
                def getsampling(path):
                    if path is None:
                        return tree.attrs['seed'], self.opts.samples
                    ds = tree[path]
                    return ds.attrs['seed'], len(ds['bestfits'])
            else:
                def getsampling(path):
                    s = self.opts.samples
                    if path is not None:
                        s = min(tree[path].attrs.get("samples", s), s)
                    return randomseed(), s
        else:
            pointset = PointSet.fromgrids(self.opts)
        return pointset, getsampling

    def initrandomizers(self):
        randomizers = {}
        for randentry in self.opts.randomize:
            pname, randdesc = randentry[0], randentry[1:]
            assert pname in self.opts.pullparams
            randomizers[self.env.pars[pname]] = self.makerandomizer(randdesc)
        return randomizers

    def initoutput(self, pointset, minimizers):
        if self.opts.segments:
            _, cur_seg = self.opts.segments
            dirname, basename = os.path.split(self.opts.output)
            try:
                base, suffix = basename.split('.')
            except ValueError:
                # output name contains multiple '.'
                spl = basename.split('.')
                base, suffix = '.'.join(spl[:-1]), spl[-1]
            new_basename = "{}_{}.{}".format(base, cur_seg, suffix)
            output = os.path.join(dirname, new_basename)
            f = PointTree(self.env, output, "w")
        else:
            f = PointTree(self.env, self.opts.output, "w")
        f.params = pointset.params
        f.attrs["allparams"] = [p.name() for p in self.allparams]
        f.attrs["minimizers"] = [n for p in self.minprops for n in p.names]
        # f.attrs["contexts"] = [m.idstring() for m in minimizers]
        # f.attrs["toymc"] = self.opts.toymc
        return f

    def initminimizers(self, pointset):
        self.minprops = []
        self.nchi2 = 0
        for fitid, minimizer in enumerate(self.opts.minimizers):
            depends = [par for par in minimizer.freepars() if minimizer.affects(par)]
            if set(depends).intersection(pointset.params):
                local = False
            else:
                local = True
            idxes = [self.paramidx[par] for par in minimizer.pars]
            nctx = 1
            nres = 1
            names = [str(fitid)]
            proprs = MinimizerProperties(local=local, idxes=idxes, nres=nres,
                                         names=names, fitvars=minimizer.freepars())
            minimizer.tolerance = self.opts.tolerance
            self.minprops.append(proprs)
            self.nchi2 += nres

    def makedata(self, nsamples):
        types = {}
        if not self.single:
            types.update({
                'ids': ((), 'i'),
            })
        types.update({
            'chi2s': ((self.nchi2,), 'f8'),
            'fcparams': ((self.nchi2, len(self.allparams)), 'f8'),
        })
        return {name: np.full((nsamples,)+shape, np.nan, dtype)
                for name, (shape, dtype) in types.items()}

    def store(self, f, path, datasets, parvalues, seed=None):
        pars = np.full(len(self.allparams), np.nan, 'f8')
        for pname, v in parvalues.items():
            pars[self.paramidx[pname]] = v
        f.touch(path).attrs['parvalues'] = pars
        for name, data in datasets.items():
            if self.single:
                assert data.shape[0] == 1
                data = data.reshape(data.shape[1:])
                if name == 'chi2s':
                    name = 'datafit'
            f.touch(path).create_dataset(name, data=data)
        if seed is not None:
            f.touch(path).attrs["seed"] = seed
        return f

    def maketoyprediction(self):
        if not self.opts.toymc:
            return
        for param, randomizer in self.randomizers.items():
            param.set(randomizer(self.toymc.random))
        self.toymc.nextSample()

    def dofits(self, minimizers, data, idx, sampleid, mask=repeat(True)):
        fitresults = [(None, None)]*len(minimizers)
        failed = False
        for fitid, (enabled, minimizer) in enumerate(zip(mask, minimizers)):
            if not enabled:
                continue
            res = minimizer.fit()
            props = self.minprops[fitid]

            if self.opts.verbose>1:
                minimizer.PrintResults()
            if res is None:
                failed = True
                break
#                or not res.success:
            fitresults[fitid] = (res, [])
        if failed:
            print(res)
            print("minimization failed")
            return False
        resid = -1
        for fitid, (enabled, (res, allres)) in enumerate(zip(mask, fitresults)):
            props = self.minprops[fitid]
            if not enabled:
                resid += props.nres
                continue
            itres = chain([res], allres)
            for resid, ctxres in islice(enumerate(itres, resid+1), props.nres):
                if 'chi2s' in data:
                    data['chi2s'][idx][resid] = ctxres.fun
                for parid, v in zip(self.minprops[fitid].idxes, ctxres.x):
                    data['fcparams'][idx][resid][parid] = v
        if 'ids' in data:
            data['ids'][idx] = sampleid
        return True

    def computeparams(self, parvalues):
        pullmin = self.opts.pullminimizer
        if not pullmin:
            return parvalues
        with self.env.pars.save(pullmin.pars):
            res = pullmin.fit()
        assert res.success
        return dict(parvalues.items(), **dict(zip(pullmin.pars, res.x)))

    def setminparams(self, envs, pdict):
        for pname, v in pdict.items():
            for env in envs:
                par = env.pars.min.GetPar(pname)
                if par:
                    par.SaveValue("@min.default", v)

    def printfit(self, path, idx, chi2s=None):
        if chi2s is not None and self.opts.verbose:
            if self.single:
                print(path, ' '.join(chi2s.astype(str)))
            else:
                print(path, idx, ' '.join(chi2s.astype(str)))
        if self.single:
            return
        nprint = 50
        if self.nwait == nprint or chi2s is None:
            t = time.time()
            msg = "{} fits, {:.2f} s, path: {}"
            print(msg.format(self.nwait, t-self.tlast, path))
            self.nwait = 0
        if self.nwait == 0:
            self.tlast = time.time()
        self.nwait += 1

    def fcscan(self, pointset, getsampling, minimizers, outfile):
        #  for path, values in tqdm.tqdm(pointset.iterpathvalues()):
        if self.opts.segments:
            nseg, cur_seg = self.opts.segments
            if self.opts.verbose:
                print("Scanning {} segments out of {}".format(cur_seg, nseg))
        else:
            nseg, cur_seg = None, None
            print('Scanning over {} out of {} segments'.format(cur_seg, nseg))
        for path, values in pointset.iterpathvalues(nseg, cur_seg):
            seed, nsamples = getsampling(path)

            if self.toymc:
                import ROOT as R
                R.GNA.Random.seed(seed)
                self.toymc.reset()
            data = self.makedata(nsamples)
            parvalues = dict(zip(pointset.params, values))
            with self.env.pars.update(parvalues):
                parvalues = self.computeparams(parvalues)

            idx = 0
            sampleid = -1
            while idx < nsamples:
                with self.env.pars.update(parvalues):
                    self.maketoyprediction()
                    sampleid += 1
                    if self.dofits(minimizers, data, idx, sampleid):
                        self.printfit(path, idx, data['chi2s'][idx])
                        idx += 1
            self.printfit(path, idx)
            self.store(outfile, path, data, parvalues)

    def h0scan(self, pointset, getsampling, minimizers, outfile):
        seed, nsamples = getsampling(None)

        if self.toymc:
            self.toymc.seed(seed)
        sdata = self.makedata(nsamples)
        data = {}
        for path, values in pointset.iterpathvalues():
            data[path] = self.makedata(nsamples)
        parvalues = self.computeparams({})
        # self.setminparams(envs, parvalues)

        samplefitmask = np.array([not mprop.local for mprop in self.minprops])
        sampleresmask = np.hstack([(not props.local,)*props.nres
                                   for props in self.minprops])

        idx = 0
        sampleid = -1
        path = None
        while idx < nsamples:
            self.maketoyprediction()
            sampleid += 1
            # self.setminparams(envs, parvalues)
            if not self.dofits(minimizers, sdata, idx, sampleid,
                               mask=samplefitmask):
                continue
            for path, values in pointset.iterpathvalues():
                # self.setminparams(envs, dict(zip(pointset.params, values)))
                with self.env.pars.update(parvalues):
                    if not self.dofits(minimizers, data[path], idx, sampleid,
                                       mask=np.logical_not(samplefitmask)):
                        break
                mask = sampleresmask
                for dsname in ['chi2s', 'fcparams']:
                    if dsname not in data[path]:
                        continue
                    data[path][dsname][idx][mask] = sdata[dsname][idx][mask]
                assert not np.isnan(data[path]['chi2s'][idx]).any()
                self.printfit(path, idx, data[path]['chi2s'][idx])
            else:
                idx += 1
        self.printfit(path, idx)
        for path, values in pointset.iterpathvalues():
            self.store(outfile, path, data[path], parvalues)
        outfile.attrs["seed"] = seed

    def run(self):
        if self.opts.toymc:
            if not self.opts.toymc_type:
                raise ValueError("please specify --toymc-type with --toymc")
            samples_type = self.opts.toymc_type
        else:
            samples_type = 'grid'
        if self.opts.samples is None:
            self.opts.samples = 1
            self.single = True
        else:
            self.single = False
        pointset, getsampling = self.points()

        allparams = set(pointset.params)
        for minimizer in self.opts.minimizers:
            allparams.update(minimizer.pars)
        self.allparams = allparams
        self.paramidx = {par: i for i, par in enumerate(self.allparams)}

        self.initminimizers(pointset)
        minimizers = self.opts.minimizers
        self.randomizers = self.initrandomizers()

        self.toymc = self.opts.toymc
        outfile = self.initoutput(pointset, minimizers)

        self.tstarted = time.time()
        self.nwait = 0
        self.tlast = 0
        if samples_type == 'grid':
            self.fcscan(pointset, getsampling, minimizers, outfile)
        elif samples_type == 'static':
            self.h0scan(pointset, getsampling, minimizers, outfile)

        return True

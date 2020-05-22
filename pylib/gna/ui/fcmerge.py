# -*- coding: utf-8 -*-


from gna.ui import basecmd
import numpy as np
import h5py
from itertools import chain
import glob
from collections import defaultdict

from gna.pointtree import PointTree

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        super(cmd, cls).initparser(parser, env)
        parser.add_argument('--fcscan', dest='fcscans', type=str, nargs='+', required=True)
        parser.add_argument('--output', type=str, required=True)
        parser.add_argument('--exclude', type=str)

    def merge(self, mergedds, ds):
        def addds(dsset, name, obj):
            if isinstance(obj, h5py.Dataset):
                dsset.add(name)

        dsnames = set()
        ds.visititems(lambda name, obj: addds(dsnames, name, obj))
        if not dsnames:
            print("skipping empty group", ds)
            return
        mergednames = set()
        mergedds.visititems(lambda name, obj: addds(mergednames, name, obj))
        if mergednames and dsnames != mergednames:
            msg = "conflicting group contents: %r; expected %r"
            raise Exception(msg % (dsnames, mergednames))
        added = None
        for dsname in dsnames:
            if dsname in mergednames:
                shape = ds[dsname].shape
                dtype = ds[dsname].dtype
                mergedshape = mergedds[dsname].shape
                mergeddtype = mergedds[dsname].dtype
                if shape[1:] != mergedshape[1:]:
                    msg = "%r: conflicting shape: %r; expected %r"
                    raise Exception(msg % (dsname, shape[1:], mergedshape[1:]))
                if dtype != mergeddtype:
                    msg = "%r: conflicting dtype: %r; expected %r"
                    raise Exception(msg % (dsname, dtype, mergeddtype))
                offset = mergedshape[0]
                n = ds[dsname].shape[0]
                if added is None:
                    added = n
                elif added != n:
                    msg = "%r: conflicting ds len: %d; expected %d"
                    raise Exception(msg % (dsname, n, added))
                mergedds[dsname].resize(offset + n, axis=0)
                mergedds[dsname][offset:] = ds[dsname][:]
            else:
                added = ds[dsname].shape[0]
                maxshape = [None]
                maxshape.extend(ds[dsname].shape[1:])
                mergedds.create_dataset(dsname, data=ds[dsname],
                                        maxshape=maxshape)
        return added

    def getmergedattr(self, ds, name):
        if name not in ds.attrs:
            if name == 'entries':
                return [ds["ids"].shape[0]]
            return None
        attr = ds.attrs[name]
        if not attr.shape:
            return [attr]
        else:
            return list(attr)

    def mergeattrs(self, attrs, tree, same=[], merged=[]):
        def equal(a, b):
            eq = (a == b)
            if isinstance(eq, bool):
                if not eq and a is b:
                    return True
                return eq
            try:
                eq |= np.isnan(a) & np.isnan(b)
                eq |= np.allclose(a, b, rtol=1e-6, atol=0)
            except TypeError:
                pass
            return eq.all()

        assert not set(same).intersection(merged)
        for attr in tree.attrs:
            if attr not in same and attr not in merged:
                msg = "unknown attribute %r"
                raise Exception(msg % attr)
        for attr in same:
            if attr in attrs:
                if not equal(attrs[attr], tree.attrs.get(attr)):
                    print(attrs[attr])
                    print(tree.attrs.get(attr))
                    msg = "%s: conflicting values for %r"
                    raise Exception(msg % (fname, attr))
            else:
                attrs[attr] = tree.attrs.get(attr)
        for attr in merged:
            cattr = self.getmergedattr(tree, attr)
            if attr in attrs:
                if attrs[attr] is None and cattr is not None:
                    msg = "%s: no attribute %r"
                    raise Exception(msg % (fname, attr))
                if attrs[attr] is not None and cattr is None:
                    msg = "%s: unexpected attribute %r"
                    raise Exception(msg % (fname, attr))
                if cattr is not None:
                    attrs[attr].extend(cattr)
            else:
                attrs[attr] = cattr
        return attrs

    def run(self):
        parnames = None
        fcmerged = PointTree(self.opts.output, "w")

        attrs = {}
        dsattrs = defaultdict(dict)
        paths = set()
        cnt = defaultdict(int)
        for fname in chain(*[glob.glob(fmask) for fmask in self.opts.fcscans]):
            try:
                tree = PointTree(fname)
            except IOError:
                print("ignoring", fname)
                continue
            attrs = self.mergeattrs(attrs, tree,
                                    same=['allparams', 'params', 'minimizers',
                                          'contexts', 'toymc'],
                                    merged=['seed'])
            for path, values, ds in tree.iterall():
                paths.add(path)
                mergedds = fcmerged.touch(path)
                dsattrs[path] = self.mergeattrs(dsattrs[path], ds,
                                                same=['parvalues'],
                                                merged=['seed', 'entries'])
                cnt[path] += self.merge(mergedds, ds)
            print("processed", fname)
            tree.close()

        for path in paths:
            mergedds = fcmerged[path]
            totalentries = sum(dsattrs[path]["entries"])
            if totalentries != cnt[path]:
                msg = "%s: invalid entries count %d, expected %d"
                raise Exception(msg % (path, cnt[path], totalentries))
            print("{:20}: {} entries".format(path, cnt[path]), end=' ')
            seeds = dsattrs[path].get('seed')
            if seeds is not None:
                print(", seeds:", seeds)
                if len(set(seed)) != len(seed):
                    msg = "%s: non-unique seeds"
                    raise Exception(msg % ds.name)
            else:
                print("")
            for attr, value in dsattrs[path].items():
                if value is not None:
                    mergedds.attrs[attr] = value

        for attr, value in attrs.items():
            if value is not None:
                fcmerged.attrs[attr] = value

        return True

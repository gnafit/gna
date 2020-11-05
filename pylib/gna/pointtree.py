import h5py
import numpy as np
import cppyy

class PointTree(object):
    def __init__(self, env, container, mode='r'):
        self.env = env
        if isinstance(container, str):
            self.root = h5py.File(container, mode)
        else:
            self.root = container
        self.attrs = self.root.attrs
        try:
            self._params = [env.pars[pname] for pname in self.root.attrs["params"]]
        except KeyError:
            self._params = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.root.close()

    def __getitem__(self, key):
        return self.root[key]

    def __setitem__(self, key, value):
        self.root[key] = value

    def __contains__(self, path):
        return path in self.root

    def close(self):
        return self.root.close()

    @property
    def params(self):
        return self._params

    @params.setter
    def params(self, params):
        self._params = []
        for param in params:
            if isinstance(param, (str, cppyy.gbl.std.string)):
                self._params.append(self.env.pars[str(param)])
            else:
                self._params.append(param)
        self.root.attrs["params"] = ['.'.join([p.ns.path, p.name()]) for p in params]

    def touch(self, node):
        path = self.path(node)
        try:
            return self.root[path]
        except KeyError:
            return self.root.create_group(path)

    def values(self, node):
        if isinstance(node, str):
            path = node
        elif isinstance(node, h5py.Group):
            path = node.name
        else:
            raise Exception("unknown pointtree node: `%s'" % repr(node))
        return [p.cast(str(x)) for p, x in zip(self._params, [_f for _f in path.rsplit('/') if _f])]

    def path(self, values):
        if isinstance(values, str):
            return values
        if isinstance(values, cppyy.gbl.std.string):
            return str(values)
        return '/'.join([str(x) for x in values])

    def iter(self, depth=None):
        f = self.root
        params = self.root.attrs["params"]
        if depth is None:
            depth = len(params)
        assert depth <= len(params)
        its = [iter([])]*depth
        groups = [None]*depth

        its[0], groups[0] = iter(f), f

        while True:
            try:
                yield groups[-1][next(its[-1])]
            except StopIteration:
                if len(its) == 1:
                    break
                for i in reversed(list(range(len(its)-1))):
                    try:
                        groups[i+1] = groups[i][next(its[i])]
                    except StopIteration:
                        if i == 0:
                            raise StopIteration()
                        continue
                    for j in range(i+1, len(its)-1):
                        its[j] = iter(groups[j])
                        groups[j+1] = groups[j][next(its[j])]
                    its[-1] = iter(groups[-1])

    def iterall(self, depth=None):
        for grp in self.iter(depth=depth):
            yield (str(grp.name.strip('/')), self.values(grp.name), grp)

    def iterpathvalues(self, depth=None):
        for grp in self.iter(depth=depth):
            yield (str(grp.name.strip('/')), self.values(grp.name))

    def itervalues(self, depth=None):
        for grp in self.iter(depth=depth):
            yield self.values(grp.name)

    def iterpaths(self, depth=None):
        for grp in self.iter(depth=depth):
            yield str(grp.name.strip('/'))

    @property
    def grids(self):
        gridset = [set() for p in self.params]
        for values in self.itervalues():
            for i, v in enumerate(values):
                gridset[i].add(v)
        return [np.array(sorted(gset)) for gset in gridset]

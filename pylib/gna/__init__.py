import os.path

from load import ROOT as R

def datapath(fname):
    basedir = os.path.join(os.path.dirname(__file__), "../../data")
    return os.path.join(basedir, fname)

class ConstructorsWrapper(object):
    def __init__(self):
        from gna import constructors
        self.constructors = constructors

    def __getattr__(self, name):
        ret = None
        try:
            ret=getattr(self.constructors, name)
        except AttributeError:
            pass

        if ret is None:
            ret = self.__findname(name)

        if not isinstance(ret, str):
            setattr(self, name, ret)
        return ret

    def __findname(self, name):
        try:
            return getattr(R, name)
        except AttributeError:
            raise Exception('Do not know GNAObject '+name)

constructors = ConstructorsWrapper()

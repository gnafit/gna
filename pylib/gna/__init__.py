import os.path

from load import ROOT as R

def datapath(fname):
    basedir = os.path.join(os.path.dirname(__file__), "../../data")
    return os.path.join(basedir, fname)

class ConstructorsWrapper(object):
    def __init__(self):
        from gna import constructors
        self.__constructors = constructors
        self.__templates = R.GNA.GNAObjectTemplates
        self.__chain = ( self.__findwrapper, self.__findtemplate, self.__findclass )
        self.__notfound=['notfound']

    def __getattr__(self, name):
        ret = self.__notfound
        for finder in self.__chain:
            try:
                # print('try', finder, name)
                ret, save = finder(name)
            except AttributeError:
                # print('  fail')
                pass
            else:
                # print('  found')
                break

        if ret is self.__notfound:
            raise Exception('Do not know GNAObject '+name)

        if save and not isinstance(ret, str):
            setattr(self, name, ret)

        return ret

    def __findwrapper(self, name):
        return getattr(self.__constructors, name), True

    def __findtemplate(self, name):
        template = self.__templates.__getattr__(name+'T')
        cls = template(self.__constructors.current_precision)
        return cls, False

    def __findclass(self, name):
        return getattr(R, name), True

constructors = ConstructorsWrapper()

from __future__ import print_function
from tools.dictwrapper import DictWrapper, DictWrapperPrinter
import pytest
from collections import OrderedDict

def test_dictwrapper_01():
    dw = DictWrapper({})

    assert not dw
    assert len(dw)==0

def test_dictwrapper_02():
    dw = DictWrapper(dict(a=1))

    assert dw
    assert len(dw)==1

def test_dictwrapper_03():
    d = dict(a=1, b=2, c=3)
    dw = DictWrapper(d)

    assert dw.get('a')==1
    assert dw.get('b')==2
    assert dw.get('c')==3
    assert dw.get('d')==None
    assert dw.get('d.e')==None

    assert tuple(dw.keys())==('a','b','c')
    assert tuple(dw.iterkeys())==('a','b','c')

@pytest.mark.parametrize('split', [None, '.'])
def test_dictwrapper_03(split):
    dct = dict(a=1, b=2, c=3, d=dict(e=4), f=dict(g=dict(h=5)))
    dct['z.z.z'] = 0
    print(dct)
    dw = DictWrapper(dct, split=split)

    #
    # Test self access
    #
    assert dw.get(()).unwrap() is dct
    assert dw[()].unwrap() is dct

    #
    # Test wrapping
    #
    assert isinstance(dw.get('d'), DictWrapper)
    assert isinstance(dw.get(('f', 'g')), DictWrapper)

    #
    # Test get tuple
    #
    assert dw.get(('d', 'e'))==4
    assert dw.get(('d', 'e1')) is None
    assert dw.get(('f', 'g', 'h'))==5
    try:
        dw.get(('z', 'z', 'z'))
        assert False
    except KeyError:
        pass

    #
    # Test getitem tuple
    #
    assert dw[('d', 'e')]==4
    try:
        dw[('d', 'e1')]
        assert False
    except KeyError:
        pass
    assert dw[('f', 'g', 'h')]==5

    try:
        dw[('z', 'z', 'z')]
        assert False
    except KeyError:
        pass

    #
    # Test get split
    #
    if split:
        assert dw.get('d.e')==4
    else:
        assert dw.get('d.e') is None

    if split:
        try:
            dw.get('z.z.z')
            assert False
        except KeyError:
            pass
    else:
        assert dw.get('z.z.z')==0

    #
    # Test getitem split
    #
    try:
        assert dw['d.e']==4
        assert split is not None
    except KeyError:
        pass

    try:
        assert dw['f.g.h']==5
        assert dw[('f.g', 'h')]==5
        assert split is not None
    except KeyError:
        pass

    if split:
        try:
            dw['z.z.z']
            assert False
        except KeyError:
            pass
    else:
        assert dw['z.z.z']==0

    #
    # Test contains
    #
    assert 'a' in dw
    assert not 'a1' in dw
    assert 'd' in dw

    #
    # Test contains tuple
    #
    assert ('d', 'e') in dw
    assert not ('k', 'e') in dw
    assert ('f', 'g', 'h') in dw
    assert ('f.g.h' in dw) == bool(split)
    assert ('z.z.z' in dw) == bool(not split)

    #
    # Test parents
    #
    g = dw.get(('f', 'g'))
    assert g.parent().parent() is dw

    #
    # Test children
    #
    m=dw.child(('k', 'l', 'm'))
    assert dw.get(('k', 'l', 'm')).unwrap() is m.unwrap()

    #
    # Test recursive setitem
    #
    dw[('k', 'l', 'm', 'n')] = 5
    try:
        dw.child(tuple('klmn'))
        assert False
    except KeyError:
        pass
    assert dw.get(('k', 'l', 'm', 'n')) == 5

    dw[('o.l.m.n')] = 6
    assert dw['o.l.m.n'] == 6
    if not split:
        assert dw.unwrap()['o.l.m.n'] == 6

    #
    # Test attribute access
    #
    assert dw._.a==1
    assert dw._.b==2
    assert dw._.c==3
    assert dw._.d.e==4
    assert dw._.f.g.h==5

    dw._.f.g.h=6
    assert dw._.f.g.h==6
    assert dw._._ is dw


def test_dictwrapper_04_visitor():
    dct = OrderedDict([('a', 1), ('b', 2), ('c', 3), ('d', dict(e=4)), ('f', dict(g=dict(h=5)))])
    dct['z.z.z'] = 0
    dw = DictWrapper(dct)

    keys0 = (('a',) , ('b',) , ('c',) , ('d', 'e'), ('f', 'g', 'h'), ('z.z.z', ))
    values0 = (1, 2, 3, 4, 5, 0)

    keys = tuple(dw.walkkeys())
    values = tuple(dw.walkvalues())
    assert keys==keys0
    assert values==values0

    class Visitor(object):
        keys, values = (), ()
        def __call__(self, k, v):
            self.keys+=k,
            self.values+=v,
    v = Visitor()
    dw.visit(v)
    assert v.keys==keys0
    assert v.values==values0

def test_dictwrapper_05_visitor():
    dct = OrderedDict([('a', 1), ('b', 2), ('c', 3), ('d', dict(e=4)), ('f', dict(g=dict(h=5)))])
    dct['z.z.z'] = 0
    dw = DictWrapper(dct)

    dw.visit(DictWrapperPrinter())

def test_dictwrapper_06_inheritance():
    dct = OrderedDict([('a', 1), ('b', 2), ('c', 3), ('d', dict(e=4)), ('f', dict(g=dict(h=5, i=6)))])
    dct['z.z.z'] = 0

    class DictWrapperA(DictWrapper):
        def count(self):
            return len(tuple(self.walkitems()))

        def depth(self):
            return max([len(k) for k in self.walkkeys()])

    dw = DictWrapperA(dct, split='.')
    assert dw.count()==7
    assert dw['d'].count()==1
    assert dw['f'].count()==2
    assert dw['f.g'].count()==2
    assert dw._.f._.count()==2

    assert dw.depth()==3
    assert dw['d'].depth()==1
    assert dw['f'].depth()==2

def test_dictwrapper_07_delete():
    dct = OrderedDict([('a', 1), ('b', 2), ('c', 3), ('d', dict(e=4)), ('f', dict(g=dict(h=5)))])
    dct['z.z.z'] = 0
    dw = DictWrapper(dct)

    assert 'a' in dw
    del dw['a']
    assert 'a' not in dw

    assert ('d', 'e') in dw
    del dw[('d', 'e')]
    assert ('d', 'e') not in dw

    assert ('f', 'g', 'h') in dw
    del dw._.f.g.h
    assert ('f', 'g', 'h') not in dw
    assert ('f', 'g') in dw

def test_dictwrapper_08_create():
    dct = OrderedDict([('a', 1), ('b', 2), ('c', 3), ('d', dict(e=4)), ('f', dict(g=dict(h=5)))])
    dct['z.z.z'] = 0
    dw = DictWrapper(dct, split='.')

    dw._('i.k').l=3
    assert dw._.i.k.l==3


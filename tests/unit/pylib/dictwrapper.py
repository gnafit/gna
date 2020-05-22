from __future__ import print_function
from tools.dictwrapper import DictWrapper
import pytest

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

@pytest.mark.parametrize('split', [None, '.'])
def test_dictwrapper_03(split):
    dct = dict(a=1, b=2, c=3, d=dict(e=4), f=dict(g=dict(h=5)))
    dw = DictWrapper(dct)

    assert dw.get(()).unwrap() is dct
    assert dw[()].unwrap() is dct
    assert isinstance(dw.get('d'), DictWrapper)
    assert isinstance(dw.get(('f', 'g')), DictWrapper)
    assert dw.get('d.e') is None
    assert dw.get(('d', 'e'))==4
    assert dw[('d', 'e')]==4
    try:
        assert dw['d.e']==4
        assert split is None
    except KeyError:
        pass
    assert dw.get(('d', 'e1')) is None
    try:
        dw[('d', 'e1')]
        assert False
    except KeyError:
        pass
    assert dw.get(('f', 'g', 'h'))==5
    assert dw[('f', 'g', 'h')]==5
    try:
        assert dw['f.g.h']==5
        assert dw[('f.g', 'h')]==5
        assert split is None
    except KeyError:
        pass

    assert 'a' in dw
    assert not 'a1' in dw
    assert 'd' in dw
    assert ('d', 'e') in dw
    assert not ('k', 'e') in dw
    assert ('f', 'g', 'h') in dw

    g = dw.get(('f', 'g'))
    assert g.parent().parent() is dw

    m=dw.child(('k', 'l', 'm'))
    assert dw.get(('k', 'l', 'm')).unwrap() is m.unwrap()

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



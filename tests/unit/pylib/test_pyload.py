from tools.pydict_load import pydict_load
from tools.cfg_load import cfg_load
from scipy import constants

def test_pydict_load_01():
    filename = 'tests/unit/pylib/test_pyload_input1.txt'
    d=pydict_load(filename, verbose=True)

    assert d.pop('proton_mass')==constants.proton_mass
    assert (d.pop('array')==[1,2,3]).all()
    assert not d

def test_pydict_load_02():
    filename = 'tests/unit/pylib/test_pyload_input2.txt'
    d=pydict_load(filename, init_globals=dict(percent=0.01), verbose=True)

    assert d.pop('full')==1.0
    assert d.pop('half')==0.5
    assert d.pop('string')=='abc'
    assert not d

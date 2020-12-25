from tools.yaml import yaml_load_file
from tools.cfg_load import cfg_load

def test_yamlload_01():
    filename = 'tests/unit/pylib/test_yaml_input1.yaml'
    d=yaml_load_file(filename)

    assert list(d.keys())==['lst', 'float', 'int', 'string']
    assert d.pop('lst')==['a', 'b', 3]
    assert d.pop('float')==1.5
    assert d.pop('int')==2
    assert d.pop('string')=='test'
    assert not d

def test_yamlload_02():
    filename = 'tests/unit/pylib/test_yaml_input1.yaml'
    d=cfg_load(filename)

    assert list(d.keys())==['lst', 'float', 'int', 'string']
    assert d.pop('lst')==['a', 'b', 3]
    assert d.pop('float')==1.5
    assert d.pop('int')==2
    assert d.pop('string')=='test'
    assert not d


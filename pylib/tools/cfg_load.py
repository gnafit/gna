from tools.yaml import yaml_load_file
from tools.pydict_load import pydict_load

def cfg_load(filename, verbose=False):
    """Load *.yaml or *.py file as dictionary"""
    if verbose:
        print('Loading dictionary from:', filename)

    if filename.endswith('.yaml'):
	return yaml_load_file(filename)
    elif filename.endswith('.py'):
	return pydict_load(filename)

    raise ValueError('Invalid configuration file type: '+filename)

def cfg_parse(filename_or_dict, verbose=False):
    """Load *.yaml or *.py file as dictionary. If the argument is dictionary - return it."""
    if isinstance(filename_or_dict, str):
	return cfg_load(filename_or_dict)

    return filename_or_dict

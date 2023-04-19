import pickle
import yaml
from gna.env import env

class FitExtractor:
    '''A class adaptor to read fit results from either from:
        - env.future["fitresult"]
        - YAML file saved with save-yaml
        - Pickle file saved with save-pickle
        '''
    def __init__(self, fit_store):
        self.fit_results = None
        self.fit_store = fit_store
        try:
            self.read()
        except Exception as e:
            raise ValueError(f'Failed to read the data from {fit_store}') from e

    def read(self):
        handle = self.fit_store
        if handle.endswith('.yml') or handle.endswith('.yaml'):
            self._read_yaml(handle)
        elif handle.endswith('.pkl'):
            self._read_pickle(handle)
        else:
            self._read_env(handle)
        return self.fit_results

    def _read_env(self, handle):
        self.fit_results = env.future['fitresult'][handle]

    def _read_pickle(self, handle):
        with open(handle, 'rb') as f:
            loaded = pickle.load(f, encoding='latin1')['fitresult']
            assert len(loaded)==1, f"Too many fit-like values in {handle}"
            for result in loaded.values():
                self.fit_results = result

    def _read_yaml(self, handle):
        with open(handle, 'r') as f:
            loaded = yaml.load(f, Loader=yaml.Loader)['fitresult']
            assert len(loaded)==1, f"Too many fit-like values in {handle}"
            for result in loaded.values():
                self.fit_results = result

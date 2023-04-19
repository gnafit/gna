from load import ROOT as R
import numpy as N
import gna.constructors as C
from gna.bundle import TransformationBundle
from gna.configurator import StripNestedDict
from schema import Schema, Or, Optional, Use, And
from tools.dictwrapper import DictWrapper

class selective_ratio_v01(TransformationBundle):
    """Selective ratio
    ```selective_ratio[i,...](a(), b())```
    computes a ratio `a/b` for indices `i,...`
    only for the indices in configuration list. Otherwise it simply yields `a`.

    Indices:
        - major: the one, to which the filter is applied
        - minor: cloning
    """
    def __init__(self, *args, **kwargs):
        TransformationBundle.__init__(self, *args, **kwargs)
        self.vcfg = self._validator.validate(StripNestedDict(self.cfg))

    # Configuration options
    _validator = Schema({
            'bundle': object,
            'name': str,                                                      # - name of the function
            'substring_skip': str,                                            # - pattern, skip ratio if pattern is found in major string
            Optional('substring_keep', default=None): str,                    # - pattern, keep ratio (overrides substring_skip) if pattern is found in major string
            Optional('labelfmt_ratio', default='ratio {autoindex}'): str,     # - label format of the Ratio
            Optional('labelfmt_view', default='view {autoindex}'): str,       # - label format for the View
            Optional('broadcast', default=False): bool                        # - whether to use broadcasting ratio
            # Optional('verbose', default=0): Or(int, And(bool, Use(int)))    # - verbosity level
        })

    @staticmethod
    def _provides(cfg):
        return (), (cfg['name'],)

    def build(self):
        self.objects = {}
        objects = DictWrapper(self.objects)

        substring_skip = self.vcfg['substring_skip']
        substring_keep = self.vcfg['substring_keep']
        name = self.vcfg['name']
        for itmajor in self.nidx_major:
            strmajor = itmajor.current_format()
            if substring_skip in strmajor:
                if substring_keep and substring_keep in strmajor:
                    make_node = self.make_ratio
                else:
                    make_node = self.make_view
            else:
                make_node = self.make_ratio

            for itminor in self.nidx_minor:
                it = itminor+itmajor
                make_node(name, it)

    def make_ratio(self, name, it):
        if self.vcfg['broadcast']:
            Ratio = C.RatioBC
        else:
            Ratio = C.Ratio
        ratio = Ratio(labels=it.current_format(self.vcfg['labelfmt_ratio']))
        input1=ratio.ratio.top
        input2=ratio.ratio.bottom
        output=ratio.single()

        self.set_input(name, it, input1, argument_number=0)
        self.set_input(name, it, input2, argument_number=1)
        self.set_output(name, it, output)

        self.objects[('ratio',)+tuple(it.current_values())]=ratio


    def make_view(self, name, it):
        view = C.View(labels=it.current_format(self.vcfg['labelfmt_view']))
        input1=view.view.inputs.data
        output=view.view.view

        self.set_input(name, it, input1, argument_number=0)
        self.set_input(name, it, (), argument_number=1)
        self.set_output(name, it, output)

        self.objects[('view',)+tuple(it.current_values())]=view

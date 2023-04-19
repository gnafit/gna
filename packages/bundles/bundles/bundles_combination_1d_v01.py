import gna.constructors as C
from gna.bundle import TransformationBundle
from gna.configurator import StripNestedDict
from tools.schema import *

from gna.bundle.bundle import get_bundle_by_cfg
from gna.configurator import NestedDict

class bundles_combination_1d_v01(TransformationBundle):
    """The bundle splits the index (1 item) into several ranges and uses different bundles to populate each range

    Notes:
        - Only 1 major index is supported!
        - No check on the consistency of the provided variables/outputs is done!

    Configuration:
        - bundles                 - dictionary with (slice_label, {label: cfg}) pairs with configuration for nested bundles
        - slices                  - dictionary with pairs (label, [index variants]) to specify which major index values are handled by which bundle
                                    a (label, 'rest') pair may be used to tell that a bundle handles all the rest index values
        - permit_empty_rest=False - if True tells the bundle not to issue an exception in case rest happens to be empty
    """
    def __init__(self, *args, **kwargs):
        TransformationBundle.__init__(self, *args, **kwargs)
        self.check_nidx_dim(1, 1, 'major')

        self.vcfg = self._validator.validate(StripNestedDict(self.cfg))
        self.configure()

    _validator = Schema({
        'bundle': object,
        'bundles': {
            str: { # slice label
                str: { # bundle label
                    'bundle': object,
                    str: object
                    }
                }
            },
        'slices': {
            str: Or((str,), [str], 'rest')
            },
        Optional('permit_empty_rest', default=False): bool,
        Optional('permit_unprocessed_rest', default=False): bool
        })

    @staticmethod
    def _provides(cfg):
        pars, outputs = set(), set()
        for slicelabel, cfgs in cfg['bundles'].items():
            for label, ccfg in cfgs.items():
                BundleClass = get_bundle_by_cfg(ccfg)
                cpars, coutputs = BundleClass.provides(ccfg)

                pars.update(cpars)
                outputs.update(coutputs)

        return tuple(pars), tuple(outputs)

    def configure(self):
        self.bundles = {}
        values_to_process = [idx.current_values()[0] for idx in self.nidx_major]

        permit_empty_rest = self.vcfg['permit_empty_rest']
        for slicelabel, cfgs in self.vcfg['bundles'].items():
            self.bundles[slicelabel] = {}

            # Determine indices to process on a current stage
            slc = self.vcfg['slices'][slicelabel]
            if slc=='rest':
                if not values_to_process:
                    if permit_empty_rest:
                        permit_empty_rest=False
                        continue
                    raise self.exception('No indicess left for the "rest"')
                values_to_process_now = tuple(values_to_process)
            else:
                values_to_process_now = slc

            for v in values_to_process_now: values_to_process.remove(v)

            # Initialize bundles
            for label, cfg in cfgs.items():
                # Find the class for the bundle
                BundleClass = get_bundle_by_cfg(cfg)
                cfg = NestedDict(cfg.copy())
                bundlecfg = cfg['bundle']
                bundlecfg.setdefault('major', self.vcfg['bundle']['major'])

                # Select indices
                nidx_major = self.nidx_major.clone()
                # Override variants in a quite hackish way
                list(nidx_major.indices.values())[0].variants = list(values_to_process_now)

                bundlecfg['nidx'] = nidx_major + self.nidx_minor

                self.bundles[slicelabel][label] = BundleClass(cfg, context=self.context)

        if values_to_process and not self.vcfg['permit_unprocessed_rest']:
            raise self.exception(f'Not all indicess processed. Leftovers: {values_to_process!s}')

    def build(self):
        for slicebundles in self.bundles.values():
            for bundle in slicebundles.values():
                bundle.build()

    def define_variables(self):
        for slicebundles in self.bundles.values():
            for bundle in slicebundles.values():
                bundle.define_variables()

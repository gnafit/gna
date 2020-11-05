
from load import ROOT as R
from gna.bundle.bundle import *

class expression_v01(TransformationBundle):
    def __init__(self, *args, **kwargs):
        TransformationBundle.__init__(self, *args, **kwargs)

        self.cfg.setdefault('verbose', False)
        # Configuration:
        #   - expr           = ''/['']      - expression(s)
        #   - lib            = {}           - names
        #   - bundles        = {}           - bundle configurations
        #   - verbose        = False/True/2 - verbose mode

        for name, cfg in self.cfg.bundles.items():
            preprocess_bundle_cfg(cfg)

    @staticmethod
    def _provides(cfg):
        return (), ()

    def build(self):
        from gna.expression.expression_v01 import Expression_v01, ExpressionContext_v01
        self.expression = Expression_v01(self.cfg.expr, self.nidx)

        if self.cfg.verbose:
            print(self.expression.expressions)

        self.expression.parse()
        self.expression.guessname(self.cfg.lib, save=True)

        if self.cfg.verbose>1:
            self.expression.tree.dump(True)

        self.expr_context = ExpressionContext_v01(self.cfg.bundles, ns=self.namespace,
                                                  inputs=self.context.inputs,
                                                  outputs=self.context.outputs)

        self.expression.build(self.expr_context)

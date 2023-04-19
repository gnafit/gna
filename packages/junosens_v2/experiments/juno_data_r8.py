from gna.experiments.fake import exp as fake

class exp(fake):
    """\
    Data loader for JUNO to read cross check data
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.opts.sources=[self.opts.input]
        self.guess_object()

    @classmethod
    def initparser(cls, parser, namespace):
        parser.add_argument('input', help='input root filename')
        parser.add_argument('-r', '--root', default='spectra', help='root path')
        parser.add_argument('-f', '--forward', nargs='+', default=[], help='Forward the data to a set of inputs from the future env', metavar='path')
        parser.add_argument('-B', '--no-bkg', action='store_true', help='Disable background')
        parser.add_argument('-v', '--verbose', default=0, action='count', help='verbosity')

    def guess_object(self):
        fname = self.opts.input.lower()

        def contains_any(*keys):
            return any(key in fname for key in keys)

        if contains_any('hongzhao', 'sysu'):
            if self.opts.no_bkg:
                name = 'case2_det_20keV'
            else:
                name = 'case3_bkg_20keV'
        elif contains_any('jinnan', 'ihep'):
            if self.opts.no_bkg:
                name = 'Case2'
            else:
                name = 'Case3'
        elif contains_any('maxim', 'dubna'):
            if self.opts.no_bkg:
                name = 'Case2_det'
            else:
                name = 'Case3_bkg'
        else:
            raise Exception('Could not guess the file type')

        self.opts.take=[name]

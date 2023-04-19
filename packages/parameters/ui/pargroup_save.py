"""Save values of a group of parameters"""

from gna.ui import basecmd
from env.lib.cwd import update_namespace_cwd

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('groups', nargs='+', help='parameter groups to work with')
        parser.add_argument('-o', '--output', required=True, help='file to store the output (yaml/pkl)')
        parser.add_argument('-v', '--verbose', action='count', default=0, help='increase verbosity level')

    def init(self):
        self.namespaces=[]
        groups_loc = self.env.future['parameter_groups']

        output = dict()
        for groupname in self.opts.groups:
            group = groups_loc[groupname]
            for k, v in group.items():
                output[k]=v.value()

        if self.opts.verbose:
            print(f'Save parameters groups {self.opts.groups} to {self.opts.output}')

            if self.opts.verbose>1:
                import pprint
                pprint.pprint(output)

        update_namespace_cwd(self.opts, 'output')
        if self.opts.output.endswith('.pkl'):
            import pickle
            with open(self.opts.output, 'wb') as ofile:
                pickle.dump(output, ofile, pickle.HIGHEST_PROTOCOL)
        elif self.opts.output.endswith('.yaml'):
            import yaml
            with open(self.opts.output, 'w') as ofile:
                ofile.write(yaml.dump(output, sort_keys=False))
        else:
            raise Exception(f'Invalid output file format of {self.opts.output}, need yaml/pkl')

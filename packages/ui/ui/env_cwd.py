"""Set GNA working directory."""

from gna.ui import basecmd
from env.lib.cwd import set_cwd, set_prefix, get_processed_paths

class cmd(basecmd):
    @classmethod
    def initparser(cls, parser, env):
        parser.add_argument('cwd', nargs='?', help='CWD to set')
        parser.add_argument('-p', '--prefix', help='Prefix')
        parser.add_argument('-d', '--print', '--dump', action='store_true', help='Print all the processed paths')

    def init(self):
        if self.opts.cwd:
            set_cwd(self.opts.cwd)
            print('CWD:', self.opts.cwd)

        if self.opts.prefix:
            set_prefix(self.opts.prefix)
            print('Prefix:', self.opts.prefix)

    def run(self):
        if self.opts.print:
            print('List of saved files:')
            tuple(map(lambda s: print('    -', s), get_processed_paths()))

    __tldr__ =  """\
                The module sets the working directory. It also checks that directory exists and is writable.
                If the directory is missing it is created with all the intermediate folders.

                Set the current working directory to 'output/test-cwd':
                ```sh
                ./gna -- env-cwd output/test-cwd
                ```
                From this moment all the output files will be saved to 'output/test-cwd'.

                An arbitrary prefix may be prepended to the filenames with `-p` option:
                ```sh
                ./gna -- env-cwd output/test-cwd -p prefix-
                ```

                At the end of the execution, the list of processed paths may be printed to stdout with `-d`:
                ```sh
                ./gna \
                    -- env-cwd output/test-cwd -p prefix-  \
                    -- cmd-save cmd.sh  \
                    -- env-cwd -d
                ```
                The `cmd-save` will save the command to the 'output/test-cwd/prefix-cmd.sh' file.
                The saved files will be printed to stdout.

                The following UI commands respect the CWD:
                - I/O
                    * `cmd_save`
                    * `save_pickle`
                    * `save_root`
                    * `save_yaml`
                - plotting:
                    * `graphviz_v1`
                    * `mpl_v1`
                """


"""Plot a graph following all the connections starting from a given output."""

from gna.ui import basecmd, append_typed, qualified
from gna.env import env, PartNotFoundError
import pygraphviz as G
import ROOT as R
from env.lib.cwd import update_namespace_cwd

from gna.graphviz import GNADot


class cmd(basecmd):
    undefined = dict()

    @classmethod
    def initparser(cls, parser, env):
        def observable(path):
            #
            # Future spectra location
            #
            try:
                return env.future['spectra'][path]
            except KeyError:
                pass

            # To be deprecated spectra location
            nspath, name = path.split('/')
            try:
                return env.ns(nspath).observables[name]
            except KeyError:
                raise PartNotFoundError("observable", path)

        parser.add_argument('plot', default=[],
                            metavar=('DATA',),
                            action=append_typed(observable))
        parser.add_argument('-J', '--no-joints', action='store_false', dest='joints', help='disable joints')
        parser.add_argument('--print-sum', action='store_true', help='add sums')
        parser.add_argument('--subgraph', action='store_true', help='enable subgraphs')
        parser.add_argument('-s', '--splines', help='splines option [dot]')
        parser.add_argument('-o', '--output', nargs='+', default=[], dest='outputs', help='output dot/pdf/png file')
        parser.add_argument('-O', '--stdout', action='store_true', help='output to stdout')
        parser.add_argument('-E', '--stderr', action='store_true', help='output to stderr')
        parser.add_argument('-n', '--namespace', '--ns', default=cls.undefined, nargs='?', help='use <namespace> to read parameters', metavar='namespace')
        parser.add_argument('--option', nargs=2, action='append', dest='options', default=[], help='AGraph kwargs key value pair')

        parser.add_argument_group(title='filters', description='Options to filter nodes')
        parser.add_argument('-i', '--include-only', '--include', nargs='+', help='Pattersn to be included (exclusive)')

    def init(self):
        update_namespace_cwd(self.opts, 'outputs')
        head = self.opts.plot[0]

        kwargs = dict(self.opts.options, joints=self.opts.joints, print_sum=self.opts.print_sum)
        kwargs.setdefault('rankdir', 'LR')
        kwargs['subgraph']=self.opts.subgraph
        kwargs['include_only']=self.opts.include_only
        if self.opts.splines:
            kwargs['splines']=self.opts.splines

        if self.opts.namespace is not self.undefined:
            kwargs['namespace']=env.globalns(self.opts.namespace)

        graph = GNADot( head, **kwargs )


        for output in self.opts.outputs:
            print( 'Write graph to:', output )

            if output.endswith('.dot'):
                graph.write(output)
            else:
                graph.layout(prog='dot')
                graph.draw(output)

        if self.opts.stdout:
            graph.write()

        if self.opts.stderr:
            import sys
            graph.write( sys.stderr )

    __tldr__ =  """\
                The modules creates a [graphviz](https://graphviz.org) representation of a GNA graph.
                It is able to save it to an image file, pdf or png.

                The module requires a reference to the output and a name of the output file, provided after an option `-o`.

                Save the graph for the minimization setup to the file 'output/graphviz-example.pdf':
                ```sh
                ./gna \\
                    -- gaussianpeak --name peak_MC --nbins 50 \\
                    -- gaussianpeak --name peak_f  --nbins 50 \\
                    -- ns --name peak_MC --print \\
                          --set E0             values=2    fixed \\
                          --set Width          values=0.5  fixed \\
                          --set Mu             values=2000 fixed \\
                          --set BackgroundRate values=1000 fixed \\
                    -- ns --name peak_f --print \\
                          --set E0             values=2.5  relsigma=0.2 \\
                          --set Width          values=0.3  relsigma=0.2 \\
                          --set Mu             values=1500 relsigma=0.25 \\
                          --set BackgroundRate values=1100 relsigma=0.25 \\
                    -- pargroup minpars peak_f -vv -m free \\
                    -- pargroup covpars peak_f -vv -m constrained \\
                    -- dataset-v1  peak --theory-data peak_f.spectrum peak_MC.spectrum -vv \\
                    -- analysis-v1 analysis --datasets peak -p covpars -v \\
                    -- stats stats --chi2 analysis \\
                    -- graphviz peak_f.spectrum -o output/graphviz-example.pdf
                ```
                In case an extension '.dot' is used the graph will be saved to a readable DOT file.

                The variables may be added to the plot by providing an option `--ns`, which may optionally be followed
                by a namespace name to limit the number of processed parameters.

                Save the graph for the minimization setup and parameters to the file 'output/graphviz-parameters-example.pdf':
                ```sh
                ./gna \\
                    -- gaussianpeak --name peak_MC --nbins 50 \\
                    -- gaussianpeak --name peak_f  --nbins 50 \\
                    -- ns --name peak_MC --print \\
                          --set E0             values=2    fixed \\
                          --set Width          values=0.5  fixed \\
                          --set Mu             values=2000 fixed \\
                          --set BackgroundRate values=1000 fixed \\
                    -- ns --name peak_f --print \\
                          --set E0             values=2.5  relsigma=0.2 \\
                          --set Width          values=0.3  relsigma=0.2 \\
                          --set Mu             values=1500 relsigma=0.25 \\
                          --set BackgroundRate values=1100 relsigma=0.25 \\
                    -- pargroup minpars peak_f -vv -m free \\
                    -- pargroup covpars peak_f -vv -m constrained \\
                    -- dataset-v1  peak --theory-data peak_f.spectrum peak_MC.spectrum -vv \\
                    -- analysis-v1 analysis --datasets peak -p covpars -v \\
                    -- stats stats --chi2 analysis \\
                    -- graphviz peak_f.spectrum -o output/graphviz-parameters-example.pdf --ns
                ```

                The module respects the CWD, which is set by `env-cwd`.

                Requires: [pygraphviz](https://pygraphviz.github.io).
                See also: `env-cwd`.
                """

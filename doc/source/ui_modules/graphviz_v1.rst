graphviz_v1
"""""""""""

The modules creates a [graphviz](https://graphviz.org) representation of a GNA graph. It is able to save it to an image file, pdf or png.


**Positional arguments**

    * ``plot`` -- name of observable to plot the graph for


**Options**

    * ``-J, --no-joints`` -- disable joints

    * ``--print-sum`` -- add sums

    * ``--subgraph`` -- enable subgraphs

    * ``-s, --splines`` -- splines option [dot]

    * ``-o, --output`` -- define path of output dot/pdf/png file

    * ``-O, --stdout`` -- output to stdout

    * ``-E, --stderr`` -- output to stderr

    * ``-n, --namespace, --ns`` -- use given namespace to read parameters

    * ``--option`` -- AGraph kwargs key value pair

    * ``-i, --include-only, --include`` -- define patterns to be included (exclusive)


**Examples**

    * Save the graph for the minimization setup to the file 'output/graphviz-example.pdf':

          .. code-block:: bash
 
              ./gna \
                  -- gaussianpeak --name peak_MC --nbins 50 \
                  -- gaussianpeak --name peak_f  --nbins 50 \
                  -- ns --name peak_MC --print \
                      --set E0             values=2    fixed \
                      --set Width          values=0.5  fixed \
                      --set Mu             values=2000 fixed \
                      --set BackgroundRate values=1000 fixed \
                  -- ns --name peak_f --print \
                      --set E0             values=2.5  relsigma=0.2 \
                      --set Width          values=0.3  relsigma=0.2 \
                      --set Mu             values=1500 relsigma=0.25 \
                      --set BackgroundRate values=1100 relsigma=0.25 \
                  -- pargroup minpars peak_f -vv -m free \
                  -- pargroup covpars peak_f -vv -m constrained \
                  -- dataset-v1  peak --theory-data peak_f.spectrum peak_MC.spectrum -vv \
                  -- analysis-v1 analysis --datasets peak -p covpars -v \
                  -- stats stats --chi2 analysis \
                  -- graphviz peak_f.spectrum -o output/graphviz-example.pdf

      In case an extension '.dot' is used the graph will be saved to a readable DOT file.


    * Save the graph for the minimization setup and parameters to the file 'output/graphviz-parameters-example.pdf':

          .. code-block:: bash
 
              ./gna \
                  -- gaussianpeak --name peak_MC --nbins 50 \
                  -- gaussianpeak --name peak_f  --nbins 50 \
                  -- ns --name peak_MC --print \
                      --set E0             values=2    fixed \
                      --set Width          values=0.5  fixed \
                      --set Mu             values=2000 fixed \
                      --set BackgroundRate values=1000 fixed \
                  -- ns --name peak_f --print \
                      --set E0             values=2.5  relsigma=0.2 \
                      --set Width          values=0.3  relsigma=0.2 \
                      --set Mu             values=1500 relsigma=0.25 \
                      --set BackgroundRate values=1100 relsigma=0.25 \
                  -- pargroup minpars peak_f -vv -m free \
                  -- pargroup covpars peak_f -vv -m constrained \
                  -- dataset-v1  peak --theory-data peak_f.spectrum peak_MC.spectrum -vv \
                  -- analysis-v1 analysis --datasets peak -p covpars -v \
                  -- stats stats --chi2 analysis \
                  -- graphviz peak_f.spectrum -o output/graphviz-parameters-example.pdf --ns


The module respects the CWD, which is set by *env-cwd*.

Requires: [pygraphviz](https://pygraphviz.github.io).

See also: *env-cwd*.

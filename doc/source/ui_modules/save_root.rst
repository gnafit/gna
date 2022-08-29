save_root
"""""""""

The module saves the paths provided as arguments to an output ROOT file, provided after `-o` option.

The outputs that should be saved should be converted via *env-data-root* module.

The module is similar to the modules *save-yaml* and *save-pickle*.


**Positional arguments**

    * ``paths`` -- define paths to save

**Options**

    * ``-o, --output`` -- define output file name

    * ``-v, --verbose`` -- define verbosity level


**Examples**

Write the data, collected in the 'output' to the file 'output.root':

.. code-block:: bash
 
    ./gna \
        -- gaussianpeak --name peak --nbins 50 \
        -- env-data-root -c spectra output \
        -- env-data-root -s spectra.peak -g fcn.x fcn.y output.fcn_graph \
        -- env-print -l 40 \
        -- save-root output -o output.root

See also: *env-data*, *save-yaml*, *save-pickle*.
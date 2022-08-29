save_pickle
"""""""""""

The module saves the paths provided as arguments to an output pickle file, provided after `-o` option.

If the outputs should be saved, the data should be converted via *env-data* module.

The pickle is a binary readable and works fast. It should be preferred over *save-yaml* for the large data.

The module is similar to the modules *save-yaml* and *save-root*.


**Positional arguments**

    * ``paths`` -- define paths to save

**Options**

    * ``-o, --output`` -- define output file name

    * ``-v, --verbose`` -- define verbosity level


**Examples**

Write the data, collected in the 'output' to the file 'output.pkl':

.. code-block:: bash
 
    ./gna \
        -- gaussianpeak --name peak --nbins 50 \
        -- env-data -c spectra output '{note: extra information}' -vv \
        -- env-print -l 40 \
        -- save-pickle output -o output.pkl

See also: *env-data*, *save-yaml*, *save-root*.
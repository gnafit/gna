save_yaml
"""""""""

The module saves the paths provided as arguments to an output YAML file, provided after `-o` option. 

If the outputs should be saved, the data should be converted via *env-data* module. 

The YAML is human readable and fits to the purposes of saving a small data samples, such as fit results or small histograms or graphs.

The module is similar to the modules *save-pickle* and *save-root*.


**Positional arguments**

    * ``paths`` -- define paths to save

**Options**

    * ``-o, --output`` -- define output file name

    * ``-v, --verbose`` -- define verbosity level


**Examples**

Write the data, collected in the 'output' to the file 'output.yaml':

.. code-block:: bash
 
    ./gna \
        -- gaussianpeak --name peak --nbins 5 \
        -- env-data -c spectra.peak output '{note: extra information}' -vv \
        -- env-print -l 40 \
        -- save-yaml output -o output.yaml

In this example we have reduced the number of bins in order to improve readability of the 'output.yaml'.

See also: *env-data*, *save-pickle*, *save-root*.
exp
"""

The dataset-v1 module loads an experiment definition that provides observables and parameters:


**Positional arguments**

    * ``experiment`` -- experiment to load 

    * ``expargs`` -- arguments defined in the experiment file

**Options**

    * ``--ns`` -- define namespace

    * ``-L, --list-experiments`` -- print list of available experiments (*only if argument ``experiment`` is empty*)
    
    * ``-h, --help`` -- print help 


**Examples**

.. code-block:: bash

    ./gna \
        -- exp --ns juno junotao_v06h -vv --binning-tao var-20 --lsnl-square

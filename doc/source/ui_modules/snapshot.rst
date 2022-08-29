snapshot
""""""""

Creates a snapshot of an observable

**Positional arguments**

    * ``name_in`` -- defines thename of input observable 

    * ``name_out`` -- defines the name of output observable 

**Options**

    * ``--ns`` -- defines the namespace

    * ``-H, --hidden`` -- make the output hidden (not observable)

    * ``-l, --label`` -- snapshot node label

**Examples**

Make snapshot of 'juno.final' observable in namespace 'juno' to 'asimov_juno' in namespace 'juno':

.. code-block:: bash

    ./gna \
        -- exp --ns juno junotao_v06h -vv --binning-tao var-20 --lsnl-square \
        -- snapshot juno.juno.final juno.asimov_juno

   
    
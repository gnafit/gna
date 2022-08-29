nmo-set
"""""""

Switch neutrino mass ordering while keeping particular mass splitting the same.


**Positional arguments**

    * ``pmns`` -- namespace with pmns parameters

**Options**

    * ``-t, --toggle`` -- switch the mass ordering
    
    * ``-s, --set``  -- set the mass ordering

        + choices: *normal* or *inverted*


    * ``- k, --keep-splitting`` -- define the mass splitting to keep while switching ordering

        + choices: *13*, *23*, *avg* or *ee*
        + default: *ee*    


    * ``--value``  -- set the particular mass splitting value

    * ``-v, --verbose`` -- define verbosity level


**Examples**

    * Initialize experiment and toggle NMO while keeping the mass splitting :math:`\Delta m_{23}^2`:

        .. code-block:: bash

            ./gna \
                -- exp --ns juno junotao_v06h -vv --binning-tao var-20 --lsnl-square \
                -- nmo-set juno.pmns -t -k 23 \


    * Initialize experiment and set NMO to 'inverted' and set default mass splitting :math:`\Delta m_{ee}^2` to a value of 0.0024707: 

        .. code-block:: bash

            ./gna \
                -- exp --ns juno junotao_v06h -vv --binning-tao var-20 --lsnl-square \
                -- nmo-set juno.pmns --set inverted --value 0.0024707 -vv



ns
""

The module manages parameters and namespaces.


**Options**

    * ``-n, --name, --ns`` -- define the namespace to work with

    * ``--new`` -- create one or more new namespaces

    * ``--push`` -- push listed namespaces to current view

    * ``--pop`` -- pop listed namespaces out of current view

    * ``--route`` -- route namespaces

        + positional arguments: *NAMESPACE1* *NAMESPACE2*


    * ``--loadset`` -- load set of paramters to namespace

        + positional arguments: *NAMESPACE* *PARSET*


    * ``--define`` -- define parameter with list of arguments

        + positional arguments: *PAR NAME* *ARG1* *ARG2* ...


    * ``--set`` -- set options of parameter

        + positional arguments: *PAR NAME* *OPT1* *OPT2* ...


    * ``--sigma`` -- set sigma of parameter

        + positional arguments: *PAR NAME* *SIGMA*


    * ``--central`` -- set central value of parameter

        + positional arguments: *PAR NAME* *CENTRAL*


    * ``--value`` -- set parameter to given value

        + positional arguments: *PAR NAME* *VALUE*


    * ``--fix`` -- set paramter fixed

    * ``--covariance`` -- compute covariance matrix of given parameters

        + positional arguments: *MATRIX NAME* *PAR1* *PAR2* ...


    * ``-o, --output`` -- define path of YAML file to dump variables to

    * ``-p, --print`` -- print namespace

    * ``--label-length`` -- define label length

    * ``--print-long`` -- do not strip long lists


**Parameter options**

    * ``value`` -- current value

    * ``values`` -- current value + central value

    * ``central`` -- central value

    * ``sigma`` -- sigma

    * ``relsigma`` -- relative sigma

    * ``fixed`` -- parameter fixed

    * ``free`` -- parameter free

    * ``pop`` -- pop parameter to previous value

    * ``push`` -- push parameter current value



**Example**

Set the parameters in the namespace 'peak':

    .. code-block:: bash
    
        ./gna \
            -- gaussianpeak --name peak \
            -- ns --name peak --print \
                --set E0             values=2.5  free \
                --set Width          values=0.3  relsigma=0.2 \
                --set Mu             values=1500 fixed \
                --set BackgroundRate values=1100 relsigma=0.25 \
            -- ns --name peak --print



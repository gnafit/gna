gaussianpeak
""""""""""""

This module initializes an example model of a gaussian peak with a flat background.


**Options**

    * ``--name`` -- define the namespace (**required**)

    * ``--npeaks`` -- define the number of peaks

        + default: 1


    * ``--Emin`` -- define the minimal energy value

        + default: 0


    * ``--Emax`` -- define the maximal energy value

        + default: 5


    * ``--nbins`` -- define the number of energy bins

        + default: 200


    * ``--order`` -- define the order of the Gauss-Legendre integrator for each bin

        + default: 8


    * ``--with-eres, --eres`` -- enable energy resolution of 3%

    * ``--print`` -- print the parameters



**Examples**

Create a gaussianpeak model 'peak' with 50 bins and print the parameters to stdout:

.. code-block:: bash

    ./gna \
        -- gaussianpeak --name peak --nbins 50 --print

   
    
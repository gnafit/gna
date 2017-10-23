Covmat_vis
^^^^^^^^^^
The `covmat_vis` module serves for the purposes of visualization of computed
covariance matrices as a block diagonal matrix.

The module accepts following command-line arguments:
    * ``--analysis`` -- the name of analysis from which covariance matrices are
      to be read.
    * ``--mask`` -- whether to mask zero values out of covariance matrix or not.
    * Mutually exclusive actions with covariance matrix:

      + ``--dump`` -- `.npz`-file where covariance matrix would be dumped under
        name `arr0`.
      + ``--show`` -- shows the plot of covariance matrix.
      + ``--savefig`` -- path to the file where figure would be saved.

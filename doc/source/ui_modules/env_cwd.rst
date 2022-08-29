env_cwd
"""""""

The module sets the working directory. It also checks that directory exists and is writable. If the directory is missing it is created with all the intermediate folders.

The cwd affects the following modules: *cmd_save*, *save_pickle*, *save_root*, *save_yaml*, *graphviz_v1*, and *mpl_v1*.

**Positional arguments**

    * ``cwd`` -- defines the path of the working directory that shall be set

**Options**

    * ``-p, --prefix`` -- defines the prefix of all files that are saved to the cwd by GNA

    * ``-d, --print, --dump`` -- print list of processed paths to console

**Examples**

    * Set the current working directory to 'output/test-cwd':

        .. code-block:: bash

            ./gna \ 
               -- env-cwd output/test-cwd
   
      From this moment all the output files will be saved to 'output/test-cwd'.

    * An arbitrary prefix may be prepended to the filenames with `-p` option:

        .. code-block:: bash

            ./gna \ 
                -- env-cwd output/test-cwd -p prefix-

    * Print list of all previously saved files:

        .. code-block:: bash

            ./gna \ 
                -- env-cwd -d
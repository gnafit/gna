cmd_save
""""""""

Saves the command line to a file. The command then may be repeated and should produce the same output.

**Positional arguments**

    * ``output`` -- define filename to which command line is saved

**Options**

    * ``-v, --verbose`` -- prints the command line to stdout

    * ``-r, --redirect`` -- add a redirection to the command

    or

    * ``-t, --tee`` -- add a redirection via tee

**Examples**

    * Save the whole command to the file 'command.sh':

        .. code-block:: bash

            ./gna \
                -- comment Initialize a gaussian peak \
                -- gaussianpeak --name peak_MC --nbins 50 \
                -- cmd-save command.sh
env_set
"""""""

The module assigns any data within env. It is needed to provide an extra information to be saved with *save-yaml* and *save-pickle*.

The module provides three ways to input data:

    1. Update env from a dictionary (nested), defined via YAML.
    2. Write a string to an address within env.
    3. Write parsed YAML to an address within env.

**Positional arguments**

    * ``update_yaml`` -- define yaml input to update the dictionary

**Options**

    * ``-r, --root`` -- define root environment

    * ``-a, --append`` -- add custom fields to the output

    * ``-y, --yaml`` -- add custom fields to the output (value parsed by yaml)


**Examples**

    * Optional argument `-r` may be used to set root address.

      Write two key-value pairs to the 'test':

        .. code-block:: bash

            ./gna \
                -- env-set -r test '{key1: string, key2: 1.0}' \
                -- env-print test

      The first value, assigned by the key 'key1' is a string 'string', the second value is a float 1.

    * The `-y` argument may be used to write a key-value pair:

        .. code-block:: bash

            ./gna \
                -- env-set -r test -y sub '{key1: string, key2: 1.0}' \
                -- env-print test

      The command does the same, but writes the key-value pairs into a nested dictionary under the key 'sub'.

    * The `-a` argument simply writes a key-value pair, where value is a string:

        .. code-block:: bash

            ./gna \
                -- env-set -r test -a key1 string \
                -- env-print test


See also: *env-print*, *env-cfg*.
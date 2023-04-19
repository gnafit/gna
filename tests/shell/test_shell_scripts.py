#!/usr/bin/env python

import os
import subprocess
from pathlib import Path
import pytest

def find_scripts(scriptspath: str = "unit_shell"):
    """Find scripts (*List[str]*) for the testing

    :param scriptspath: path to directory with scripts in tests
    :type scriptspath: str
    """
    spath = Path(__file__).parent.parent / scriptspath

    scripts = [file for file in spath.rglob("*")
               if file.is_file() and os.access(file, os.X_OK)]

    print("The following list of executable files is found:")
    print('\n'.join(f'- {str(sc)}' for sc in scripts))  # the list with scripts to testing
    return scripts

def run_script(script):
    print("Starting the test of the following script: ", script)
    sp = subprocess.Popen(
        script,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    out, err = sp.communicate(input=None)
    code = sp.returncode
    return out, err, code

@pytest.mark.parametrize("script", find_scripts())
def test_shell_scripts(script):
    """Simple test of shell scripts:
    True if the exit code of the script is 0, else False

    NOTE: If the script returns nothing, the test will pass,
          even if there are internal errors!
          Use shell strict mode!
    """
    out, err, code = run_script(script)
    print(f"Finished running the {script}:")
    print("Script output:\n", out.decode("utf-8"))
    print("Script err:\n", err.decode("utf-8"))
    print("Exit code:\n", code)

    assert code == 0, f"{script} returned non-zero exit code"

    # NOTE: more strict check: True <-> stderr is None
    # assert not err.decode("utf-8")

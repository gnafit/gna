from collections import abc
from typing import Mapping, Any
import os
import re
from pathlib import Path
from gna.configurator import NestedDict  # for legacy

try:
    from colorama import Style

    def colorize(msg: str) -> str:
        return Style.BRIGHT + msg + Style.RESET_ALL


except ImportError:

    def colorize(msg: str) -> str:
        return msg


def substitute_pathlikes(config: Mapping[Any, Any]) -> None:
    def is_path_like(path: Any) -> bool:
        if not isinstance(path, (Path, str)):
            return False
        p = path.as_posix() if isinstance(path, Path) else path
        possible_extensions = [
            ".root", ".hdf", ".hdf5", ".yml", ".yaml", ".txt",
            ".dat", ".csv", ".tsv", ".py", ".npz", ".npy",
        ]
        return any(p.endswith(ext) for ext in possible_extensions)

    def try_substitute(conf, key, it):
        if not is_path_like(it):
            return
        for path in pathes_to_data:
            tmp = path / it
            tmp_name = tmp.name
            # handling the case when pathlike is actually index
            # dependent value such as spectra_{isotope}_13.0_0.05_MeV.txt.
            # Regex below will substitute anything between {} with * so we will
            # be able to glob the its' parent folder to check that there is
            # anything that satisfy glob.
            # https://stackoverflow.com/a/40621332
            pattern = re.sub(r"\{[^{}]*\}", "*", tmp_name)
            cond = (
                tmp.exists() if pattern == tmp.name else any(tmp.parent.glob(pattern))
            )
            if cond:
                print(colorize(f"Appended {path} to {it}"))
                conf[key] = tmp.as_posix()

    pathes_to_data = os.environ.get("GNA_DATAPATH")
    if not pathes_to_data:
        return
    pathes_to_data = [Path(_) for _ in pathes_to_data.split(":")]
    for key, val in config.items():
        if isinstance(val, abc.MutableSequence):
            for idx, it in enumerate(val):
                try_substitute(val, idx, it)
        elif isinstance(val, (NestedDict, abc.MutableMapping)):
            for key, it in val.items():
                try_substitute(val, key, it)
        else:
            try_substitute(config, key, val)

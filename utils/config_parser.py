import argparse
import logging
import pathlib
from typing import Any, Dict, Tuple, Union

import yaml

__all__ = ["ConfigParser", "parse_argument"]


class ConfigParser:
    def __init__(self, config_file: pathlib.Path, args: Dict[str, Any]) -> None:
        self.data = self._read_yaml(config_file)
        for key, value in args.items():
            self._add_data(key, value)

    def _read_yaml(self, yaml_file: pathlib.Path) -> Dict:
        with open(yaml_file, "r") as f:
            data = yaml.safe_load(f)
        return data

    def _add_data(self, key: str, value: Any) -> None:
        self.data[key] = value

    def build(self, object_type: str, module, *args, **kwargs):
        submodule = kwargs.get("submodule", None)
        if submodule is not None:
            kwargs.pop("submodule")
            module_name = self.data[object_type][submodule].get("name", None)
        else:
            module_name = self.data[object_type].get("name", None)

        if module_name is None:
            return None

        if submodule is not None:
            module_args = self.data[object_type][submodule].get("args", None)
        else:
            module_args = self.data[object_type].get("args", None)
        if module_args is None:
            return getattr(module, module_name)(*args, **kwargs)

        return getattr(module, module_name)(*args, **module_args, **kwargs)

    def get_logger(self, name: str):
        verbosity = self.data["logger"]["verbosity"]
        logger = logging.getLogger(name)
        logger.setLevel(verbosity)
        logging.basicConfig(
            format="[%(levelname)s] From %(filename)s:%(funcName)s at %(asctime)s:%(msecs)03d\n>>> %(message)s",
            datefmt="%H:%M:%S",
        )

        return logger

    def __getitem__(self, name: str) -> Union[Dict, None]:
        return self.data.get(name, None)


def parse_argument() -> Tuple[pathlib.Path, Dict[str, Any]]:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", required=True, type=str, help="Configuration file"
    )
    parser.add_argument(
        "-r", "--resume", default=None, type=str, help="Path to checkpoint to load."
    )

    args = parser.parse_args()
    args_dict = vars(args)
    config_file = args_dict.pop("config")
    return (config_file, args_dict)

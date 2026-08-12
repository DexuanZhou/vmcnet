"""Derive a selected-model C config for one production hierarchy level."""

import argparse
from pathlib import Path

from vmcnet.utils import io


SPECS = {
    "L0": ((64, 8), (64,)),
    "L05": ((96, 12), (96,)),
    "L1": ((128, 16), (128, 16), (128,)),
    "L15": ((192, 16), (192, 16), (192, 16), (192,)),
    "L2": ((256, 16), (256, 16), (256, 16), (256,)),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--level", choices=tuple(SPECS), required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--output-name", default="config")
    args = parser.parse_args()

    source = Path(args.base_config).resolve()
    config = io.load_config_dict(str(source.parent), source.name)
    config.model.backflow.ndense_list = SPECS[args.level]
    io.save_config_dict_to_json(config, args.output_dir, args.output_name)


if __name__ == "__main__":
    main()

"""Create the fine-level config used by the multilevel smoke test."""

import argparse

from vmcnet.utils import io


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coarse-config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    config = io.load_config_dict(args.coarse_config.rsplit("/", 1)[0], "config.json")
    config.model.backflow.ndense_list = ((48, 12), (48, 12), (48,))
    io.save_config_dict_to_json(config, args.output_dir, "fine_config")


if __name__ == "__main__":
    main()

"""Write a complete coarse FermiNet preset before CLI flag parsing."""

import argparse

from vmcnet.train import default_config
from vmcnet.utils import io


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    config = default_config.get_default_config()
    config.model.type = "ferminet"
    config.problem.ion_pos = ((0.0, 0.0, 0.0),)
    config.problem.ion_charges = (6.0,)
    config.problem.nelec = (4, 2)
    config.model.ferminet.ndeterminants = 16
    config.model.ferminet.backflow.ndense_list = ((32, 8), (32,))
    io.save_config_dict_to_json(config, args.output_dir, "coarse_preset")


if __name__ == "__main__":
    main()

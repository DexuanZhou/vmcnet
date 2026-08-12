#!/usr/bin/env python3
import pathlib


def validate_checkpoint_epoch(checkpoint_path, internal_epoch):
    """Require VMCNet's zero-based epoch for a checkpoint named N.npz."""
    checkpoint_path = pathlib.Path(checkpoint_path)
    named_epoch = int(checkpoint_path.stem)
    expected = named_epoch - 1
    if int(internal_epoch) != expected:
        raise ValueError(
            f"{checkpoint_path.name}: internal epoch {internal_epoch}, expected {expected}"
        )
    return expected

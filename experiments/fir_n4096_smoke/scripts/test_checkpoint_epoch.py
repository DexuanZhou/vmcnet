#!/usr/bin/env python3
from checkpoint_epoch import validate_checkpoint_epoch


assert validate_checkpoint_epoch("5000.npz", 4999) == 4999
try:
    validate_checkpoint_epoch("5000.npz", 1234)
except ValueError:
    pass
else:
    raise AssertionError("unrelated internal epoch must fail")
print("PASS: 5000.npz/4999 accepted; 5000.npz/1234 rejected")

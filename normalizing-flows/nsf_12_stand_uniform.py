"""Alias wrapper for `nsf_12_stand_unform.py`.

The original filename has a typo ("unform"). This wrapper exists so you can run:
  python nsf_12_stand_uniform.py ...
"""

from nsf_12_stand_unform import parse_args, train


if __name__ == "__main__":
    train(parse_args())


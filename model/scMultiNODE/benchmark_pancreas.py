#!/usr/bin/env python
"""scMultiNODE moscot pancreas full / held-out E15.5 benchmark entry point."""
from benchmark_gastrulation import main
from dataset_protocols import PANCREAS


if __name__ == "__main__":
    main(dataset=PANCREAS)

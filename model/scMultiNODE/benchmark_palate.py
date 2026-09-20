#!/usr/bin/env python
"""scMultiNODE palate full / held-out E13.5 / held-out E14.0 entry point."""
from benchmark_gastrulation import main
from dataset_protocols import PALATE
from palate_input_audit import audit_inputs


if __name__ == "__main__":
    main(dataset=PALATE, input_auditor=audit_inputs)

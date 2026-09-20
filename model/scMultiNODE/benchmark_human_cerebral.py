#!/usr/bin/env python
"""Full-only scMultiNODE human cerebral benchmark on current CytoBridge inputs."""
from benchmark_gastrulation import main
from dataset_protocols import HUMAN_CEREBRAL
from human_cerebral_input_audit import audit_inputs


if __name__ == "__main__":
    main(dataset=HUMAN_CEREBRAL, input_auditor=audit_inputs)

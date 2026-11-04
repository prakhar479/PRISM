"""Experiment harness for "Hidden in the Leftovers" (Phase 1+). The pilot code in src/ is frozen; see REPO_MAP.md.

Pure tensor code (CPU-testable): harness.sae, harness.conditions, harness.directions, harness.metrics, harness.stats.
Model-dependent code: harness.model. Machine settings come from harness.profile; devices/dtypes from harness.device.
"""
HARNESS_VERSION = "0.1.0"

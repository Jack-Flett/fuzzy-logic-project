"""Fuzzy-logic term project: ANFIS / SOM / PSO / attention / PANFIS re-implementation."""
import os

# The models are tiny, so multi-threaded linear algebra only adds overhead (and fights with the
# worker processes of the experiment scripts). Must be set before NumPy / PyTorch are imported.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

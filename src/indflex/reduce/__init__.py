"""Step 3 of the chain: reduce detailed plant models to small linear ones (D3R, thesis Ch. 4)."""

from .d3r import (ALFParams, D3RHistory, build_alf, build_inverse_alf, fit_alf, nrmse_paper, simulate_alf,
                  train_d3r)

__all__ = ["ALFParams", "D3RHistory", "build_alf", "simulate_alf", "build_inverse_alf", "fit_alf", "train_d3r",
           "nrmse_paper"]

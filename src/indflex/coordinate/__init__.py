"""Step 4 of the chain: operate flexible resources together (VPP bidding and disaggregation)."""

from .vpp import (BidResult, HourResult, Segments, VPPCase, allocate, allocate_lp, bid, load_vpp_case, merit_order,
                  simulate_hour, to_resources)

__all__ = ["VPPCase", "load_vpp_case", "bid", "BidResult", "merit_order", "Segments", "allocate", "allocate_lp",
           "to_resources", "simulate_hour", "HourResult"]

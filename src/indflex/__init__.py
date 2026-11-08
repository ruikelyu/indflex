"""indflex: industrial load flexibility, from process models to grid decisions.

The chain (one subpackage per step of the thesis):

1. ``indflex.models``    - STN / LSTN (continuous) and RTN / cRTN (batch) models (Ch. 2)
2. ``indflex.identify``  - parameter identification from meter data, PSI (Ch. 3)
3. ``indflex.reduce``    - data-driven dimension reduction, D3R      (Ch. 4)
4. ``indflex.coordinate``- grid / aggregator optimization            (Ch. 4-5, planned)
"""

from .data import (BatchPlant, BatchStage, Mode, Plant, Stage, daily_price, list_plants, load_batch_plant,
                   load_fleet, load_plant, load_prices)
from .models import build_crtn, build_fleet, build_lstn, build_rtn, build_stn
from .solve import Result, count_vars, solve

from . import identify, reduce  # noqa: E402  (steps 2 and 3 of the chain)

__version__ = "0.1.0"

__all__ = [
    "Mode", "Plant", "Stage", "load_plant", "list_plants", "load_prices", "daily_price", "load_fleet",
    "BatchPlant", "BatchStage", "load_batch_plant",
    "build_stn", "build_lstn", "build_fleet", "build_rtn", "build_crtn", "solve", "count_vars", "Result",
]

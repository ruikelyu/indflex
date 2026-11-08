"""indflex: industrial load flexibility, from process models to grid decisions.

The chain (one subpackage per step of the thesis):

1. ``indflex.models``    - STN / LSTN / RTN / cRTN process models   (Ch. 2)
2. ``indflex.identify``  - parameter identification from meter data (Ch. 3, planned)
3. ``indflex.reduce``    - data-driven dimension reduction (D3R)     (Ch. 4, planned)
4. ``indflex.coordinate``- grid / aggregator optimization            (Ch. 4-5, planned)
"""

from .data import Mode, Plant, Stage, daily_price, list_plants, load_fleet, load_plant, load_prices
from .models import build_fleet, build_lstn, build_stn
from .solve import Result, solve

__version__ = "0.1.0"

__all__ = [
    "Mode", "Plant", "Stage", "load_plant", "list_plants", "load_prices", "daily_price", "load_fleet",
    "build_stn", "build_lstn", "build_fleet", "solve", "Result",
]

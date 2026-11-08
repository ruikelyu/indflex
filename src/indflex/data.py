"""Plant parameters and electricity prices.

All inputs live as plain text under ``data/`` at the repository root:

* ``data/plants/<name>.yaml``  - serial continuous-process plants (STN/LSTN)
* ``data/prices/pjm_rt_2022_07.csv`` - PJM real-time prices, $/MWh, hourly
* ``data/fleets/lstn_fleet_<N>.csv`` - randomized factory fleets for scaling tests
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

DATA_DIR = Path(__file__).resolve().parents[2] / "data"


@dataclass(frozen=True)
class Mode:
    """One operating point of a machine: production rate ``g`` and power ``e``."""

    g: float  # production rate, material units per hour (e.g. t/h)
    e: float  # power, kW


@dataclass(frozen=True)
class Stage:
    """A machine plus the buffer (storage) holding its output material."""

    name: str
    modes: tuple[Mode, ...]
    s_max: float  # buffer capacity
    s_init: float  # buffer level at the start of the horizon
    s_target_change: float = 0.0  # required net increase over the horizon


@dataclass(frozen=True)
class Plant:
    """A serial production line: stage k consumes the output of stage k-1.

    The first stage draws raw material without limit; the last stage's buffer
    holds the final product, whose required daily increase sets the production target.
    """

    name: str
    stages: tuple[Stage, ...]
    description: str = ""
    source: str = ""
    units: dict = field(default_factory=dict)

    @property
    def n_stages(self) -> int:
        return len(self.stages)

    def with_target(self, final_target: float) -> "Plant":
        """Return a copy with a different final-product target (material per horizon)."""
        stages = list(self.stages)
        stages[-1] = replace(stages[-1], s_target_change=final_target)
        return replace(self, stages=tuple(stages))


def load_plant(name_or_path: str | Path) -> Plant:
    """Load a plant from ``data/plants/<name>.yaml`` or an explicit YAML path."""
    path = Path(name_or_path)
    if not path.suffix:
        path = DATA_DIR / "plants" / f"{name_or_path}.yaml"
    raw = yaml.safe_load(path.read_text())
    if raw.get("kind") == "batch":
        raise ValueError(f"{path.name} is a batch plant; use load_batch_plant()")
    stages = tuple(
        Stage(
            name=s["name"],
            modes=tuple(Mode(float(m["g"]), float(m["e"])) for m in s["modes"]),
            s_max=float(s["s_max"]),
            s_init=float(s["s_init"]),
            s_target_change=float(s.get("s_target_change", 0.0)),
        )
        for s in raw["stages"]
    )
    return Plant(
        name=raw.get("name", path.stem),
        stages=stages,
        description=raw.get("description", ""),
        source=raw.get("source", ""),
        units=raw.get("units", {}),
    )


@dataclass(frozen=True)
class BatchStage:
    """One stage of a batch line (e.g. EAF, AOD, LF, CC)."""

    name: str
    power_mw: float  # nominal power while processing
    processing_min: float  # processing time of one heat
    transfer_min: float = 0.0  # time to move a heat to the next stage
    max_wait_min: float = 0.0  # longest a heat may wait before the next stage


@dataclass(frozen=True)
class BatchPlant:
    """A serial batch line processing discrete heats.

    ``power_ratio`` = (low, high) power of the flexible first stage relative to
    nominal; melting at lower power takes proportionally longer.
    """

    name: str
    stages: tuple[BatchStage, ...]
    power_ratio: tuple[float, float] = (1.0, 1.0)
    description: str = ""
    source: str = ""

    @property
    def n_stages(self) -> int:
        return len(self.stages)


def load_batch_plant(name_or_path: str | Path) -> BatchPlant:
    """Load a batch line from ``data/plants/<name>.yaml`` (``kind: batch``)."""
    path = Path(name_or_path)
    if not path.suffix:
        path = DATA_DIR / "plants" / f"{name_or_path}.yaml"
    raw = yaml.safe_load(path.read_text())
    if raw.get("kind") != "batch":
        raise ValueError(f"{path.name} is not a batch plant; use load_plant()")
    return BatchPlant(
        name=raw.get("name", path.stem),
        stages=tuple(BatchStage(**s) for s in raw["stages"]),
        power_ratio=tuple(raw.get("power_ratio", (1.0, 1.0))),
        description=raw.get("description", ""),
        source=raw.get("source", ""),
    )


def list_plants() -> list[str]:
    return sorted(p.stem for p in (DATA_DIR / "plants").glob("*.yaml"))


def load_prices(month: str = "2022-07") -> pd.DataFrame:
    """Daily price profiles as a (days x 24) DataFrame in $/MWh.

    Rows are calendar days (index = date), columns are hours 0..23.
    """
    path = DATA_DIR / "prices" / f"pjm_rt_{month.replace('-', '_')}.csv"
    df = pd.read_csv(path, parse_dates=["datetime_beginning_ept"])
    ts = df["datetime_beginning_ept"]
    table = df.assign(date=ts.dt.date, hour=ts.dt.hour).pivot_table(
        index="date", columns="hour", values="price_usd_per_mwh", aggfunc="mean"
    )
    return table


def daily_price(month: str = "2022-07", day: int = 1) -> np.ndarray:
    """Hourly price vector ($/MWh, length 24) for one day of the month (1-based)."""
    return load_prices(month).iloc[day - 1].to_numpy(dtype=float)


def load_fleet(n_factories: int) -> list[Plant]:
    """Randomized fleet of steel-powder-like factories used in the LSTN scaling test."""
    df = pd.read_csv(DATA_DIR / "fleets" / f"lstn_fleet_{n_factories}.csv")
    plants = []
    for f, df_f in df.groupby("factory"):
        stages = []
        for s, df_s in df_f.groupby("stage"):
            modes = tuple(Mode(float(r.g), float(r.e)) for r in df_s.itertuples())
            r0 = df_s.iloc[0]
            stages.append(
                Stage(
                    name=f"stage{s + 1}",
                    modes=modes,
                    s_max=float(r0.s_max),
                    s_init=float(r0.s_init),
                    s_target_change=float(r0.s_target_change),
                )
            )
        plants.append(Plant(name=f"factory{f + 1}", stages=tuple(stages)))
    return plants

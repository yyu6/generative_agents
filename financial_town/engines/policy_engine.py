"""Policy and macro-condition updates."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Dict, List

from financial_town.simulation.state import TownState


def _clamp(value: float, lower: float, upper: float) -> float:
  return max(lower, min(upper, value))


class PolicyEngine:
  """Applies scheduled policy shocks and mild macro drift."""

  def __init__(self, events: List[dict], rng: random.Random):
    self.rng = rng
    self.events_by_step: Dict[int, List[dict]] = defaultdict(list)
    for event in events:
      step = int(event.get("step", -1))
      if step >= 0:
        self.events_by_step[step].append(event)

  def run_step(self, state: TownState) -> None:
    for event in self.events_by_step.get(state.step, []):
      for key, value in event.get("policy", {}).items():
        if hasattr(state.policy, key):
          setattr(state.policy, key, value)
      for key, value in event.get("macro", {}).items():
        if hasattr(state.macro, key):
          setattr(state.macro, key, value)

    # Small stochastic drift to avoid static worlds between shocks.
    state.macro.gdp_growth = _clamp(
      state.macro.gdp_growth + self.rng.uniform(-0.002, 0.002),
      -0.08,
      0.08,
    )
    state.macro.inflation = _clamp(
      state.macro.inflation + self.rng.uniform(-0.0015, 0.0015),
      -0.01,
      0.20,
    )
    state.macro.unemployment_rate = _clamp(
      state.macro.unemployment_rate + self.rng.uniform(-0.003, 0.003),
      0.015,
      0.25,
    )
    state.macro.housing_growth = _clamp(
      state.macro.housing_growth + self.rng.uniform(-0.002, 0.002),
      -0.08,
      0.12,
    )

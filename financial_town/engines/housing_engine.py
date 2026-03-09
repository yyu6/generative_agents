"""Housing valuation and rent evolution."""

from __future__ import annotations

import random

from financial_town.simulation.state import TownState


class HousingEngine:
  def __init__(self, rng: random.Random):
    self.rng = rng

  def run_step(self, state: TownState) -> None:
    # Update each property valuation.
    for prop in state.properties.values():
      valuation_growth = (
        prop.appreciation_rate
        + state.macro.housing_growth
        + self.rng.uniform(-0.01, 0.01)
      )
      rent_growth = state.macro.inflation + self.rng.uniform(-0.004, 0.008)
      prop.market_value = round(max(40000.0, prop.market_value * (1.0 + valuation_growth)), 2)
      prop.rent_value = round(max(450.0, prop.rent_value * (1.0 + rent_growth)), 2)

    # Renters are exposed more directly to inflation.
    for household in state.households.values():
      if household.housing_mode == "rent":
        inflation_adjustment = state.macro.inflation + self.rng.uniform(-0.002, 0.01)
        household.monthly_housing_cost = round(
          max(350.0, household.monthly_housing_cost * (1.0 + inflation_adjustment)),
          2,
        )
      else:
        # Owner costs evolve slower; mostly insurance/tax maintenance.
        owner_adjustment = 0.15 * state.macro.inflation + self.rng.uniform(-0.001, 0.003)
        household.monthly_housing_cost = round(
          max(300.0, household.monthly_housing_cost * (1.0 + owner_adjustment)),
          2,
        )

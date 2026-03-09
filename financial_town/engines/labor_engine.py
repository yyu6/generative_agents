"""Labor dynamics: job loss, re-employment, promotion, and job switching."""

from __future__ import annotations

import random
from typing import Dict, List

from financial_town.models import EmploymentStatus
from financial_town.simulation.state import TownState

INDUSTRY_JOBS: Dict[str, List[str]] = {
  "healthcare": ["nurse", "medical_assistant", "billing_specialist"],
  "education": ["teacher", "school_admin", "advisor"],
  "retail": ["cashier", "sales_associate", "store_supervisor"],
  "technology": ["analyst", "software_engineer", "product_manager"],
  "finance": ["teller", "loan_officer", "risk_analyst"],
  "hospitality": ["front_desk", "cook", "service_manager"],
  "logistics": ["driver", "warehouse_specialist", "planner"],
}

TRANSITION_INDUSTRIES = list(INDUSTRY_JOBS.keys())


def _income_to_tier(annual_income: float) -> str:
  if annual_income < 35000:
    return "low"
  if annual_income < 90000:
    return "middle"
  if annual_income < 180000:
    return "upper_middle"
  return "high"


class LaborEngine:
  def __init__(self, rng: random.Random):
    self.rng = rng

  def _layoff_probability(self, state: TownState, industry: str) -> float:
    base = state.policy.layoff_rate + 0.50 * max(0.0, state.macro.unemployment_rate - 0.05)
    if industry in {"retail", "hospitality"}:
      base += 0.012
    if industry in {"technology", "finance"}:
      base -= 0.003
    return max(0.001, min(0.25, base))

  def _rehire_probability(self, state: TownState) -> float:
    base = 0.18 + (state.macro.gdp_growth * 1.6) - (state.macro.unemployment_rate * 0.8)
    return max(0.02, min(0.70, base))

  def run_step(self, state: TownState) -> None:
    for person in state.people.values():
      if person.employment_status == EmploymentStatus.EMPLOYED:
        # Layoff shock.
        if self.rng.random() < self._layoff_probability(state, person.industry):
          person.employment_status = EmploymentStatus.UNEMPLOYED
          continue

        # Promotion.
        promotion_prob = 0.018 + state.policy.promotion_boost + max(0.0, state.macro.gdp_growth * 0.3)
        if person.education_level in {"master", "professional"}:
          promotion_prob += 0.01
        if self.rng.random() < min(0.20, promotion_prob):
          raise_factor = self.rng.uniform(1.03, 1.12)
          person.annual_base_income = round(person.annual_base_income * raise_factor, 2)
          person.pay_tier = _income_to_tier(person.annual_base_income)

        # Voluntary job change.
        switch_prob = 0.015 + (0.012 if person.risk_tolerance > 0.7 else 0.0)
        if self.rng.random() < switch_prob:
          new_industry = self.rng.choice(TRANSITION_INDUSTRIES)
          person.industry = new_industry
          person.occupation = self.rng.choice(INDUSTRY_JOBS[new_industry])
          person.annual_base_income = round(
            person.annual_base_income * self.rng.uniform(0.95, 1.20), 2
          )
          person.pay_tier = _income_to_tier(person.annual_base_income)

      else:
        # Re-employment.
        if self.rng.random() < self._rehire_probability(state):
          person.employment_status = EmploymentStatus.EMPLOYED
          new_industry = self.rng.choice(TRANSITION_INDUSTRIES)
          person.industry = new_industry
          person.occupation = self.rng.choice(INDUSTRY_JOBS[new_industry])
          if person.annual_base_income <= 0:
            person.annual_base_income = round(self.rng.uniform(30000, 90000), 2)
          else:
            person.annual_base_income = round(
              person.annual_base_income * self.rng.uniform(0.90, 1.08),
              2,
            )
          person.pay_tier = _income_to_tier(person.annual_base_income)

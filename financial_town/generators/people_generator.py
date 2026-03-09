"""Synthetic population, household, housing, and loan generator.

Inspired by synthetic data projects and agent-based financial simulators:
  - TwinMarket-style market participants
  - Apache Fineract-style loan lifecycle fields
  - LLM-Economist-style policy shock experimentation
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Tuple

from financial_town.config import SimulationConfig
from financial_town.models import (
  EmploymentStatus,
  EmploymentType,
  Household,
  Loan,
  LoanType,
  MacroState,
  PayCycle,
  Person,
  PolicyState,
  Property,
)
from financial_town.simulation.state import TownState

DEFAULT_CATEGORY_PRICE_MULTIPLIERS = {
  "groceries": 1.0,
  "utilities": 1.0,
  "transport": 1.0,
  "healthcare": 1.0,
  "education": 1.0,
  "retail": 1.0,
  "entertainment": 1.0,
}

FIRST_NAMES = [
  "Ava", "Noah", "Emma", "Liam", "Mia", "Ethan", "Olivia", "Sophia",
  "Lucas", "Amelia", "Elijah", "Harper", "Mason", "Ella", "Logan",
  "Isabella", "Benjamin", "Evelyn", "James", "Charlotte",
]

LAST_NAMES = [
  "Rodriguez", "Patel", "Nguyen", "Thompson", "Jackson", "Kim", "Lopez",
  "Smith", "Brown", "Williams", "Davis", "Johnson", "Clark", "Miller",
  "Garcia", "Lee", "Hernandez", "Wilson", "Martin", "Hall",
]

EDUCATION_LEVELS = [
  "high_school",
  "associate",
  "bachelor",
  "master",
  "professional",
]

INDUSTRY_JOBS: Dict[str, List[str]] = {
  "healthcare": ["nurse", "medical_assistant", "billing_specialist"],
  "education": ["teacher", "school_admin", "advisor"],
  "retail": ["cashier", "sales_associate", "store_supervisor"],
  "technology": ["analyst", "software_engineer", "product_manager"],
  "finance": ["teller", "loan_officer", "risk_analyst"],
  "hospitality": ["front_desk", "cook", "service_manager"],
  "logistics": ["driver", "warehouse_specialist", "planner"],
}

CITY_ZONES = ["downtown", "midtown", "suburb", "industrial_edge"]

COMPANY_SIZES = ["micro", "small", "mid", "large", "enterprise"]

SPENDING_STYLES = ["frugal", "balanced", "impulsive"]

ZONE_COORD_RANGES = {
  "downtown": {"home": ((65, 95), (20, 35)), "work": ((80, 108), (28, 48)), "spend": ((84, 112), (30, 52))},
  "midtown": {"home": ((100, 128), (36, 56)), "work": ((90, 120), (34, 54)), "spend": ((94, 126), (38, 58))},
  "suburb": {"home": ((112, 135), (48, 66)), "work": ((96, 124), (34, 52)), "spend": ((106, 132), (42, 62))},
  "industrial_edge": {"home": ((45, 75), (58, 75)), "work": ((52, 80), (46, 66)), "spend": ((56, 84), (50, 70))},
}


def _choose_household_size(rng: random.Random, max_size: int) -> int:
  # Typical household size distribution with long tail.
  population = [1, 2, 3, 4, min(5, max_size)]
  weights = [0.30, 0.34, 0.18, 0.12, 0.06]
  return min(max_size, rng.choices(population, weights=weights, k=1)[0])


def _sample_zone_coord(rng: random.Random, zone: str, purpose: str) -> Tuple[int, int]:
  ranges = ZONE_COORD_RANGES.get(zone, ZONE_COORD_RANGES["midtown"])[purpose]
  return rng.randint(ranges[0][0], ranges[0][1]), rng.randint(ranges[1][0], ranges[1][1])


def _amortized_monthly_payment(principal: float, annual_rate: float, months: int) -> float:
  if months <= 0:
    return 0.0
  monthly_rate = annual_rate / 12.0
  if monthly_rate <= 0:
    return principal / float(months)
  numerator = monthly_rate * principal
  denominator = 1.0 - math.pow(1.0 + monthly_rate, -months)
  return numerator / denominator


def _income_to_tier(annual_income: float) -> str:
  if annual_income < 35000:
    return "low"
  if annual_income < 90000:
    return "middle"
  if annual_income < 180000:
    return "upper_middle"
  return "high"


def _pick_employment_type(rng: random.Random) -> EmploymentType:
  return rng.choices(
    [EmploymentType.SALARIED, EmploymentType.HOURLY, EmploymentType.SELF_EMPLOYED],
    weights=[0.58, 0.28, 0.14],
    k=1,
  )[0]


def _annual_income_for_job(
  rng: random.Random,
  employment_type: EmploymentType,
  industry: str,
) -> float:
  base_by_industry = {
    "technology": (70000, 190000),
    "finance": (55000, 160000),
    "healthcare": (50000, 140000),
    "education": (42000, 110000),
    "logistics": (38000, 90000),
    "retail": (28000, 65000),
    "hospitality": (26000, 75000),
  }
  low, high = base_by_industry[industry]
  if employment_type == EmploymentType.HOURLY:
    low *= 0.80
    high *= 0.85
  if employment_type == EmploymentType.SELF_EMPLOYED:
    low *= 0.70
    high *= 1.15
  return round(rng.uniform(low, high), 2)


def _pay_cycle_for_type(employment_type: EmploymentType, rng: random.Random) -> PayCycle:
  if employment_type == EmploymentType.SALARIED:
    return rng.choices([PayCycle.BIWEEKLY, PayCycle.MONTHLY], [0.65, 0.35], k=1)[0]
  if employment_type == EmploymentType.HOURLY:
    return rng.choices([PayCycle.WEEKLY, PayCycle.BIWEEKLY], [0.60, 0.40], k=1)[0]
  return PayCycle.MONTHLY


def _init_influence_state(people: Dict[str, Person]) -> Dict[str, Dict[str, float]]:
  out: Dict[str, Dict[str, float]] = {}
  for person_id in people:
    out[person_id] = {
      "spending_multiplier": 1.0,
      "loan_stress": 0.0,
      "fraud_susceptibility": 1.0,
    }
  return out


def _build_social_graph(
  households: Dict[str, Household],
  people: Dict[str, Person],
  rng: random.Random,
) -> Dict[str, Dict[str, List[str]]]:
  graph: Dict[str, Dict[str, List[str]]] = {person_id: {} for person_id in people}

  def add_rel(p1: str, p2: str, relation: str) -> None:
    graph.setdefault(p1, {})
    graph.setdefault(p2, {})
    graph[p1].setdefault(p2, [])
    graph[p2].setdefault(p1, [])
    if relation not in graph[p1][p2]:
      graph[p1][p2].append(relation)
    if relation not in graph[p2][p1]:
      graph[p2][p1].append(relation)

  # Family relations from shared household.
  for household in households.values():
    members = household.member_ids
    for i in range(len(members)):
      for j in range(i + 1, len(members)):
        add_rel(members[i], members[j], "family")

  # Coworkers if they share industry and are employed.
  person_list = list(people.values())
  for i in range(len(person_list)):
    for j in range(i + 1, len(person_list)):
      p1 = person_list[i]
      p2 = person_list[j]
      if p1.employment_status == EmploymentStatus.EMPLOYED and p2.employment_status == EmploymentStatus.EMPLOYED:
        if p1.industry == p2.industry and rng.random() < 0.20:
          add_rel(p1.person_id, p2.person_id, "coworker")

  # Neighbors by city zone.
  by_zone: Dict[str, List[str]] = {}
  for household in households.values():
    by_zone.setdefault(household.city_zone, [])
    by_zone[household.city_zone].extend(household.member_ids)
  for zone_members in by_zone.values():
    for i in range(len(zone_members)):
      for j in range(i + 1, len(zone_members)):
        if rng.random() < 0.05:
          add_rel(zone_members[i], zone_members[j], "neighbor")

  return graph


def _bootstrap_policy(policy_payload: dict) -> PolicyState:
  policy = PolicyState()
  for key, value in policy_payload.items():
    if hasattr(policy, key):
      setattr(policy, key, value)
  return policy


def _bootstrap_macro(macro_payload: dict) -> MacroState:
  macro = MacroState()
  for key, value in macro_payload.items():
    if hasattr(macro, key):
      setattr(macro, key, value)
  return macro


def build_initial_town(
  config: SimulationConfig,
  policy_bundle: dict,
  rng: random.Random,
) -> TownState:
  """
  Create an initial simulation state with synthetic people and financial profiles.
  """
  households: Dict[str, Household] = {}
  people: Dict[str, Person] = {}
  properties: Dict[str, Property] = {}
  loans: Dict[str, Loan] = {}

  person_counter = 1
  property_counter = 1
  loan_counter = 1

  for household_index in range(1, config.households + 1):
    household_id = f"HH-{household_index:04d}"
    household_size = _choose_household_size(rng, config.max_household_size)
    zone = rng.choices(CITY_ZONES, weights=[0.25, 0.28, 0.39, 0.08], k=1)[0]
    housing_mode = "own" if rng.random() < config.home_ownership_rate else "rent"

    base_housing = {
      "downtown": (1800, 4500),
      "midtown": (1300, 3600),
      "suburb": (1000, 2800),
      "industrial_edge": (900, 2200),
    }
    housing_cost = round(rng.uniform(*base_housing[zone]), 2)

    households[household_id] = Household(
      household_id=household_id,
      member_ids=[],
      city_zone=zone,
      housing_mode=housing_mode,
      monthly_housing_cost=housing_cost,
    )

    if housing_mode == "own":
      market_value = round(housing_cost * rng.uniform(110, 180), 2)
      rent_value = round(housing_cost * rng.uniform(0.80, 1.05), 2)
      property_id = f"PR-{property_counter:04d}"
      property_counter += 1
      properties[property_id] = Property(
        property_id=property_id,
        city_zone=zone,
        market_value=market_value,
        rent_value=rent_value,
        appreciation_rate=rng.uniform(0.002, 0.007),
        owner_household_id=household_id,
      )

    for member_index in range(household_size):
      person_id = f"P-{person_counter:05d}"
      person_counter += 1

      first_name = rng.choice(FIRST_NAMES)
      last_name = rng.choice(LAST_NAMES)
      full_name = f"{first_name} {last_name} {person_id[-3:]}"
      age = rng.randint(22, 76)
      education = rng.choices(
        EDUCATION_LEVELS,
        weights=[0.20, 0.18, 0.34, 0.20, 0.08],
        k=1,
      )[0]

      industry = rng.choice(list(INDUSTRY_JOBS.keys()))
      occupation = rng.choice(INDUSTRY_JOBS[industry])
      employment_type = _pick_employment_type(rng)
      annual_income = _annual_income_for_job(rng, employment_type, industry)
      pay_cycle = _pay_cycle_for_type(employment_type, rng)
      pay_tier = _income_to_tier(annual_income)

      unemployed = rng.random() < 0.08
      employment_status = EmploymentStatus.UNEMPLOYED if unemployed else EmploymentStatus.EMPLOYED
      if unemployed:
        annual_income = 0.0

      savings_ratio = rng.uniform(0.15, 0.60)
      starting_cash = round(rng.uniform(600, 3000), 2)
      starting_savings = round((annual_income / 12.0) * 6.0 * savings_ratio, 2)
      home_xy = _sample_zone_coord(rng, zone, "home")
      work_xy = _sample_zone_coord(rng, zone, "work")
      spend_xy = _sample_zone_coord(rng, zone, "spend")

      person = Person(
        person_id=person_id,
        full_name=full_name,
        age=age,
        family_size=household_size,
        education_level=education,
        household_id=household_id,
        employment_type=employment_type,
        employment_status=employment_status,
        industry=industry,
        occupation=occupation,
        company_size=rng.choice(COMPANY_SIZES),
        pay_cycle=pay_cycle,
        annual_base_income=annual_income,
        pay_tier=pay_tier,
        checking_balance=starting_cash,
        savings_balance=starting_savings,
        credit_score=rng.randint(560, 830),
        risk_tolerance=round(rng.uniform(0.1, 0.95), 3),
        spending_style=rng.choice(SPENDING_STYLES),
        last_gross_income=round(annual_income / 12.0, 2),
        home_x=home_xy[0],
        home_y=home_xy[1],
        work_x=work_xy[0],
        work_y=work_xy[1],
        spending_x=spend_xy[0],
        spending_y=spend_xy[1],
        home_address=f"the Ville:{full_name}'s home:main room:bed",
        work_address=f"the Ville:{industry} district:{occupation}:workstation",
        spending_address=f"the Ville:{zone}:commercial strip:shop",
        current_x=home_xy[0],
        current_y=home_xy[1],
        current_place="home",
      )

      households[household_id].member_ids.append(person_id)
      people[person_id] = person

      # Add a subset of personal liabilities.
      if age <= 42 and rng.random() < 0.22:
        principal = round(rng.uniform(4500, 42000), 2)
        rate = policy_bundle["policy"].get("central_bank_rate", 0.035) + 0.025 + rng.uniform(0.00, 0.02)
        term = rng.choice([60, 84, 120])
        monthly_payment = round(_amortized_monthly_payment(principal, rate, term), 2)
        loan_id = f"L-{loan_counter:06d}"
        loan_counter += 1
        loans[loan_id] = Loan(
          loan_id=loan_id,
          borrower_person_id=person_id,
          loan_type=LoanType.EDUCATION,
          purpose="education",
          principal=principal,
          remaining_principal=principal,
          annual_rate=round(rate, 4),
          remaining_term_months=term,
          monthly_payment=monthly_payment,
        )
        person.loan_ids.append(loan_id)

      if rng.random() < 0.12:
        principal = round(rng.uniform(1200, 11000), 2)
        rate = policy_bundle["policy"].get("central_bank_rate", 0.035) + 0.07 + rng.uniform(0.00, 0.05)
        term = rng.choice([24, 36, 48])
        monthly_payment = round(_amortized_monthly_payment(principal, rate, term), 2)
        loan_id = f"L-{loan_counter:06d}"
        loan_counter += 1
        loans[loan_id] = Loan(
          loan_id=loan_id,
          borrower_person_id=person_id,
          loan_type=LoanType.PERSONAL,
          purpose="personal_expense",
          principal=principal,
          remaining_principal=principal,
          annual_rate=round(rate, 4),
          remaining_term_months=term,
          monthly_payment=monthly_payment,
        )
        person.loan_ids.append(loan_id)

  # Add mortgages to owner households where possible.
  owner_props = [p for p in properties.values() if p.owner_household_id]
  for prop in owner_props:
    household = households[prop.owner_household_id]
    if not household.member_ids:
      continue
    owner_person_id = household.member_ids[0]
    if rng.random() < 0.80:
      principal = round(prop.market_value * rng.uniform(0.45, 0.85), 2)
      rate = (
        policy_bundle["policy"].get("central_bank_rate", 0.035)
        + policy_bundle["policy"].get("loan_rate_spread", 0.03)
        + rng.uniform(0.005, 0.02)
      )
      term = rng.choice([180, 240, 300, 360])
      monthly_payment = round(_amortized_monthly_payment(principal, rate, term), 2)
      loan_id = f"L-{loan_counter:06d}"
      loan_counter += 1
      loans[loan_id] = Loan(
        loan_id=loan_id,
        borrower_person_id=owner_person_id,
        loan_type=LoanType.HOME,
        purpose="home_purchase",
        principal=principal,
        remaining_principal=principal,
        annual_rate=round(rate, 4),
        remaining_term_months=term,
        monthly_payment=monthly_payment,
      )
      people[owner_person_id].loan_ids.append(loan_id)
      prop.mortgage_loan_id = loan_id

  social_graph = _build_social_graph(households, people, rng)
  influence_state = _init_influence_state(people)

  return TownState(
    step=0,
    scenario_name=policy_bundle.get("scenario_name", "custom"),
    people=people,
    households=households,
    properties=properties,
    loans=loans,
    policy=_bootstrap_policy(policy_bundle.get("policy", {})),
    macro=_bootstrap_macro(policy_bundle.get("macro", {})),
    social_graph=social_graph,
    influence_state=influence_state,
    consumer_price_index=1.0,
    category_price_multipliers=dict(DEFAULT_CATEGORY_PRICE_MULTIPLIERS),
  )

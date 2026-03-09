"""Single-agent world generator: one free agent + scripted population."""

from __future__ import annotations

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

ZONE_COORD_RANGES = {
  "downtown": {"home": ((65, 95), (20, 35)), "work": ((80, 108), (28, 48)), "spend": ((84, 112), (30, 52))},
  "midtown": {"home": ((100, 128), (36, 56)), "work": ((90, 120), (34, 54)), "spend": ((94, 126), (38, 58))},
  "suburb": {"home": ((112, 135), (48, 66)), "work": ((96, 124), (34, 52)), "spend": ((106, 132), (42, 62))},
  "industrial_edge": {"home": ((45, 75), (58, 75)), "work": ((52, 80), (46, 66)), "spend": ((56, 84), (50, 70))},
}

FIRST_NAMES = [
  "Olivia", "Liam", "Emma", "Noah", "Sophia", "Mason", "Ava", "Ethan", "Isabella", "Lucas",
  "Mia", "James", "Amelia", "Benjamin", "Charlotte", "Elijah", "Harper", "Logan", "Evelyn", "Daniel",
]

LAST_NAMES = [
  "Rodriguez", "Patel", "Nguyen", "Thompson", "Jackson", "Kim", "Lopez", "Smith", "Brown", "Williams",
  "Davis", "Johnson", "Clark", "Miller", "Garcia", "Lee", "Hernandez", "Wilson", "Martin", "Hall",
]

EDUCATION_POOL = ["high_school", "associate", "bachelor", "master", "professional"]


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


def _sample_zone_coord(rng: random.Random, zone: str, purpose: str) -> Tuple[int, int]:
  ranges = ZONE_COORD_RANGES.get(zone, ZONE_COORD_RANGES["midtown"])[purpose]
  return rng.randint(ranges[0][0], ranges[0][1]), rng.randint(ranges[1][0], ranges[1][1])


def _income_tier(annual_income: float) -> str:
  if annual_income < 35000:
    return "low"
  if annual_income < 90000:
    return "middle"
  if annual_income < 180000:
    return "upper_middle"
  return "high"


def _scripted_income(rng: random.Random, industry: str) -> float:
  base_by_industry = {
    "technology": (76000, 172000),
    "finance": (56000, 145000),
    "healthcare": (46000, 122000),
    "education": (39000, 96000),
    "logistics": (36000, 84000),
    "retail": (28000, 64000),
    "hospitality": (26000, 70000),
  }
  low, high = base_by_industry[industry]
  return round(rng.uniform(low, high), 2)


def _maybe_add_scripted_loan(
  rng: random.Random,
  loans: Dict[str, Loan],
  person: Person,
  loan_counter: List[int],
) -> None:
  if rng.random() > 0.35:
    return

  loan_id = f"L-SCRIPT-{loan_counter[0]:05d}"
  loan_counter[0] += 1

  loan_type = rng.choice([LoanType.PERSONAL, LoanType.EDUCATION, LoanType.INVESTMENT])
  principal = round(rng.uniform(2800, 36000), 2)
  remaining_principal = round(principal * rng.uniform(0.45, 1.0), 2)
  annual_rate = round(rng.uniform(0.055, 0.145), 4)
  remaining_term = rng.choice([12, 24, 36, 60])
  monthly_payment = round(max(90.0, principal / max(1, remaining_term) * 1.08), 2)

  loans[loan_id] = Loan(
    loan_id=loan_id,
    borrower_person_id=person.person_id,
    loan_type=loan_type,
    purpose=f"scripted_{loan_type.value}",
    principal=principal,
    remaining_principal=remaining_principal,
    annual_rate=annual_rate,
    remaining_term_months=remaining_term,
    monthly_payment=monthly_payment,
  )
  person.loan_ids.append(loan_id)


def build_single_agent_world(
  config: SimulationConfig,
  policy_bundle: dict,
  rng: random.Random,
) -> TownState:
  people: Dict[str, Person] = {}
  households: Dict[str, Household] = {}
  properties: Dict[str, Property] = {}
  loans: Dict[str, Loan] = {}

  scripted_population = max(1, int(config.scripted_population_size))

  # 1) Create one active/free agent.
  active_zone = "midtown"
  active_home = _sample_zone_coord(rng, active_zone, "home")
  active_work = _sample_zone_coord(rng, active_zone, "work")
  active_spend = _sample_zone_coord(rng, active_zone, "spend")

  active_household_id = "HH-ACTIVE-001"
  active_person_id = "P-ACTIVE-001"
  active_person_name = (config.active_agent_name or "Alex Carter").strip() or "Alex Carter"

  households[active_household_id] = Household(
    household_id=active_household_id,
    member_ids=[active_person_id],
    city_zone=active_zone,
    housing_mode="rent",
    monthly_housing_cost=1680.0,
  )

  active_agent = Person(
    person_id=active_person_id,
    full_name=active_person_name,
    age=31,
    family_size=1,
    education_level="master",
    household_id=active_household_id,
    employment_type=EmploymentType.SALARIED,
    employment_status=EmploymentStatus.EMPLOYED,
    industry="finance",
    occupation="risk_analyst",
    company_size="large",
    pay_cycle=PayCycle.BIWEEKLY,
    annual_base_income=122000.0,
    pay_tier="upper_middle",
    checking_balance=4600.0,
    savings_balance=22000.0,
    credit_score=756,
    risk_tolerance=0.59,
    spending_style="balanced",
    home_x=active_home[0],
    home_y=active_home[1],
    work_x=active_work[0],
    work_y=active_work[1],
    spending_x=active_spend[0],
    spending_y=active_spend[1],
    home_address=f"the Ville:{active_person_name}'s apartment:main room:bed",
    work_address="the Ville:Capital Bank:risk floor:desk",
    spending_address="the Ville:Main Street:coffee bar:table",
    current_x=active_home[0],
    current_y=active_home[1],
    current_place="home",
  )
  people[active_person_id] = active_agent

  # 2) Create scripted population (Truman-world style, fixed profiles per seed).
  loan_counter = [1]
  scripted_ids: List[str] = []
  for i in range(1, scripted_population + 1):
    person_id = f"P-SCRIPT-{i:03d}"
    household_id = f"HH-SCRIPT-{i:03d}"

    first_name = FIRST_NAMES[(i - 1) % len(FIRST_NAMES)]
    last_name = LAST_NAMES[(i * 3) % len(LAST_NAMES)]
    full_name = f"{first_name} {last_name} {i:03d}"

    zone = CITY_ZONES[(i + 1) % len(CITY_ZONES)]
    industry = list(INDUSTRY_JOBS.keys())[(i + 2) % len(INDUSTRY_JOBS)]
    occupation = INDUSTRY_JOBS[industry][i % len(INDUSTRY_JOBS[industry])]

    annual_income = _scripted_income(rng, industry)
    employed = rng.random() > 0.09

    home_xy = _sample_zone_coord(rng, zone, "home")
    work_xy = _sample_zone_coord(rng, zone, "work")
    spend_xy = _sample_zone_coord(rng, zone, "spend")

    households[household_id] = Household(
      household_id=household_id,
      member_ids=[person_id],
      city_zone=zone,
      housing_mode="rent" if rng.random() < 0.73 else "own",
      monthly_housing_cost=round(rng.uniform(920, 2850), 2),
    )

    person = Person(
      person_id=person_id,
      full_name=full_name,
      age=rng.randint(22, 69),
      family_size=1,
      education_level=EDUCATION_POOL[i % len(EDUCATION_POOL)],
      household_id=household_id,
      employment_type=EmploymentType.SALARIED,
      employment_status=EmploymentStatus.EMPLOYED if employed else EmploymentStatus.UNEMPLOYED,
      industry=industry,
      occupation=occupation,
      company_size=rng.choice(["small", "mid", "large", "enterprise"]),
      pay_cycle=rng.choice([PayCycle.BIWEEKLY, PayCycle.MONTHLY]),
      annual_base_income=annual_income if employed else 0.0,
      pay_tier=_income_tier(annual_income),
      checking_balance=round(rng.uniform(500, 6200), 2),
      savings_balance=round(rng.uniform(1500, 39000), 2),
      credit_score=rng.randint(590, 820),
      risk_tolerance=round(rng.uniform(0.12, 0.9), 3),
      spending_style=rng.choice(["frugal", "balanced", "impulsive"]),
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

    people[person_id] = person
    scripted_ids.append(person_id)
    _maybe_add_scripted_loan(rng, loans, person, loan_counter)

  # 3) Active agent connected to scripted world; scripted-to-scripted edges are sparse.
  social_graph: Dict[str, Dict[str, List[str]]] = {person_id: {} for person_id in people}

  def add_rel(p1: str, p2: str, relation: str) -> None:
    social_graph.setdefault(p1, {})
    social_graph.setdefault(p2, {})
    social_graph[p1].setdefault(p2, [])
    social_graph[p2].setdefault(p1, [])
    if relation not in social_graph[p1][p2]:
      social_graph[p1][p2].append(relation)
    if relation not in social_graph[p2][p1]:
      social_graph[p2][p1].append(relation)

  for npc_id in scripted_ids:
    npc = people[npc_id]
    if npc.industry == active_agent.industry:
      add_rel(active_person_id, npc_id, "coworker")
    add_rel(active_person_id, npc_id, "neighbor")

  # Sparse scripted relationships for ambient interactions.
  for idx in range(0, len(scripted_ids) - 1, 2):
    p1 = scripted_ids[idx]
    p2 = scripted_ids[idx + 1]
    add_rel(p1, p2, "coworker")
    if rng.random() < 0.30:
      add_rel(p1, p2, "neighbor")

  influence_state: Dict[str, Dict[str, float]] = {
    person_id: {
      "spending_multiplier": 1.0,
      "loan_stress": 0.0,
      "fraud_susceptibility": 1.0,
    }
    for person_id in people
  }

  return TownState(
    step=0,
    scenario_name=policy_bundle.get("scenario_name", "single_agent_world"),
    people=people,
    households=households,
    properties=properties,
    loans=loans,
    policy=_bootstrap_policy(policy_bundle.get("policy", {})),
    macro=_bootstrap_macro(policy_bundle.get("macro", {})),
    social_graph=social_graph,
    influence_state=influence_state,
    active_agent_id=active_person_id,
    scripted_agent_ids=scripted_ids,
    consumer_price_index=1.0,
    category_price_multipliers=dict(DEFAULT_CATEGORY_PRICE_MULTIPLIERS),
  )

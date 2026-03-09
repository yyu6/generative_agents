"""Hand-crafted 3-agent profile using Stanford Smallville starter names."""

from __future__ import annotations

from typing import Dict, List

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


def build_stanford_n3_town(policy_bundle: dict) -> TownState:
  households: Dict[str, Household] = {
    "HH-ISABELLA": Household(
      household_id="HH-ISABELLA",
      member_ids=["P-ISABELLA"],
      city_zone="downtown",
      housing_mode="rent",
      monthly_housing_cost=1850.0,
    ),
    "HH-MARIA": Household(
      household_id="HH-MARIA",
      member_ids=["P-MARIA"],
      city_zone="midtown",
      housing_mode="rent",
      monthly_housing_cost=1280.0,
    ),
    "HH-KLAUS": Household(
      household_id="HH-KLAUS",
      member_ids=["P-KLAUS"],
      city_zone="midtown",
      housing_mode="rent",
      monthly_housing_cost=1320.0,
    ),
  }

  people: Dict[str, Person] = {
    "P-ISABELLA": Person(
      person_id="P-ISABELLA",
      full_name="Isabella Rodriguez",
      age=34,
      family_size=1,
      education_level="master",
      household_id="HH-ISABELLA",
      employment_type=EmploymentType.SALARIED,
      employment_status=EmploymentStatus.EMPLOYED,
      industry="technology",
      occupation="product_manager",
      company_size="mid",
      pay_cycle=PayCycle.BIWEEKLY,
      annual_base_income=138000.0,
      pay_tier="upper_middle",
      checking_balance=3400.0,
      savings_balance=28000.0,
      credit_score=782,
      risk_tolerance=0.66,
      spending_style="balanced",
      home_x=72,
      home_y=14,
      work_x=95,
      work_y=27,
      spending_x=88,
      spending_y=33,
      home_address="the Ville:Isabella Rodriguez's apartment:main room:bed",
      work_address="the Ville:Tech Hub:PM floor:workstation",
      spending_address="the Ville:Hobbs Cafe:main hall:table",
      current_x=72,
      current_y=14,
      current_place="home",
    ),
    "P-MARIA": Person(
      person_id="P-MARIA",
      full_name="Maria Lopez",
      age=26,
      family_size=1,
      education_level="bachelor",
      household_id="HH-MARIA",
      employment_type=EmploymentType.SALARIED,
      employment_status=EmploymentStatus.EMPLOYED,
      industry="education",
      occupation="advisor",
      company_size="large",
      pay_cycle=PayCycle.MONTHLY,
      annual_base_income=64000.0,
      pay_tier="middle",
      checking_balance=2100.0,
      savings_balance=9800.0,
      credit_score=708,
      risk_tolerance=0.48,
      spending_style="balanced",
      home_x=123,
      home_y=57,
      work_x=117,
      work_y=50,
      spending_x=112,
      spending_y=55,
      home_address="the Ville:Dorm for Oak Hill College:Maria Lopez's room:bed",
      work_address="the Ville:Oak Hill College:advising office:desk",
      spending_address="the Ville:Dorm Commons:coffee corner:seat",
      current_x=123,
      current_y=57,
      current_place="home",
    ),
    "P-KLAUS": Person(
      person_id="P-KLAUS",
      full_name="Klaus Mueller",
      age=29,
      family_size=1,
      education_level="master",
      household_id="HH-KLAUS",
      employment_type=EmploymentType.SALARIED,
      employment_status=EmploymentStatus.EMPLOYED,
      industry="education",
      occupation="school_admin",
      company_size="large",
      pay_cycle=PayCycle.MONTHLY,
      annual_base_income=71000.0,
      pay_tier="middle",
      checking_balance=2400.0,
      savings_balance=11300.0,
      credit_score=722,
      risk_tolerance=0.57,
      spending_style="frugal",
      home_x=126,
      home_y=46,
      work_x=117,
      work_y=50,
      spending_x=120,
      spending_y=43,
      home_address="the Ville:Dorm for Oak Hill College:Klaus Mueller's room:bed",
      work_address="the Ville:Oak Hill College:admin office:desk",
      spending_address="the Ville:College Plaza:bookstore:cafe",
      current_x=126,
      current_y=46,
      current_place="home",
    ),
  }

  properties: Dict[str, Property] = {}
  loans: Dict[str, Loan] = {}

  loan_a = Loan(
    loan_id="L-ISABELLA-001",
    borrower_person_id="P-ISABELLA",
    loan_type=LoanType.INVESTMENT,
    purpose="small_business_investment",
    principal=18000.0,
    remaining_principal=12800.0,
    annual_rate=0.082,
    remaining_term_months=24,
    monthly_payment=818.0,
  )
  loan_b = Loan(
    loan_id="L-MARIA-001",
    borrower_person_id="P-MARIA",
    loan_type=LoanType.EDUCATION,
    purpose="student_loan",
    principal=32000.0,
    remaining_principal=21400.0,
    annual_rate=0.061,
    remaining_term_months=74,
    monthly_payment=352.0,
  )
  loan_c = Loan(
    loan_id="L-KLAUS-001",
    borrower_person_id="P-KLAUS",
    loan_type=LoanType.PERSONAL,
    purpose="family_support",
    principal=9000.0,
    remaining_principal=6100.0,
    annual_rate=0.109,
    remaining_term_months=21,
    monthly_payment=470.0,
  )
  loans[loan_a.loan_id] = loan_a
  loans[loan_b.loan_id] = loan_b
  loans[loan_c.loan_id] = loan_c
  people["P-ISABELLA"].loan_ids.append(loan_a.loan_id)
  people["P-MARIA"].loan_ids.append(loan_b.loan_id)
  people["P-KLAUS"].loan_ids.append(loan_c.loan_id)

  # Multi-edge social graph (family, coworkers, neighbors).
  social_graph: Dict[str, Dict[str, List[str]]] = {
    "P-ISABELLA": {
      "P-MARIA": ["neighbor"],
      "P-KLAUS": ["neighbor"],
    },
    "P-MARIA": {
      "P-ISABELLA": ["neighbor"],
      "P-KLAUS": ["family", "coworker", "neighbor"],
    },
    "P-KLAUS": {
      "P-ISABELLA": ["neighbor"],
      "P-MARIA": ["family", "coworker", "neighbor"],
    },
  }

  influence_state = {
    "P-ISABELLA": {"spending_multiplier": 1.0, "loan_stress": 0.0, "fraud_susceptibility": 1.0},
    "P-MARIA": {"spending_multiplier": 1.0, "loan_stress": 0.0, "fraud_susceptibility": 1.0},
    "P-KLAUS": {"spending_multiplier": 1.0, "loan_stress": 0.0, "fraud_susceptibility": 1.0},
  }

  return TownState(
    step=0,
    scenario_name=policy_bundle.get("scenario_name", "stanford_n3"),
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

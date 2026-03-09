"""Domain dataclasses for people, households, assets, and events."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .enums import (
  EmploymentStatus,
  EmploymentType,
  LoanStatus,
  LoanType,
  PayCycle,
  TransactionType,
)


@dataclass
class Person:
  person_id: str
  full_name: str
  age: int
  family_size: int
  education_level: str
  household_id: str

  employment_type: EmploymentType
  employment_status: EmploymentStatus
  industry: str
  occupation: str
  company_size: str
  pay_cycle: PayCycle
  annual_base_income: float
  pay_tier: str

  checking_balance: float
  savings_balance: float
  credit_score: int
  risk_tolerance: float
  spending_style: str

  loan_ids: List[str] = field(default_factory=list)
  last_gross_income: float = 0.0

  # Mobility and routine.
  home_x: int = 0
  home_y: int = 0
  work_x: int = 0
  work_y: int = 0
  spending_x: int = 0
  spending_y: int = 0
  home_address: str = ""
  work_address: str = ""
  spending_address: str = ""
  current_x: int = 0
  current_y: int = 0
  current_place: str = "home"


@dataclass
class Household:
  household_id: str
  member_ids: List[str]
  city_zone: str
  housing_mode: str  # "own" or "rent"
  monthly_housing_cost: float


@dataclass
class Property:
  property_id: str
  city_zone: str
  market_value: float
  rent_value: float
  appreciation_rate: float
  owner_household_id: Optional[str] = None
  mortgage_loan_id: Optional[str] = None


@dataclass
class Loan:
  loan_id: str
  borrower_person_id: str
  loan_type: LoanType
  purpose: str

  principal: float
  remaining_principal: float
  annual_rate: float
  remaining_term_months: int
  monthly_payment: float
  status: LoanStatus = LoanStatus.ACTIVE
  delinquent_months: int = 0


@dataclass
class PolicyState:
  income_tax_rate: float = 0.20
  payroll_tax_rate: float = 0.0765
  unemployment_replacement_rate: float = 0.45
  central_bank_rate: float = 0.035
  promotion_boost: float = 0.0
  layoff_rate: float = 0.015
  housing_subsidy: float = 0.0
  loan_rate_spread: float = 0.030
  stimulus_payment: float = 0.0
  fraud_detection_strength: float = 0.60


@dataclass
class MacroState:
  gdp_growth: float = 0.020
  inflation: float = 0.025
  unemployment_rate: float = 0.050
  housing_growth: float = 0.020


@dataclass
class Transaction:
  transaction_id: str
  step: int
  timestamp: str

  person_id: Optional[str]
  household_id: Optional[str]

  category: str
  tx_type: TransactionType
  amount: float

  fraudulent: bool = False
  note: str = ""

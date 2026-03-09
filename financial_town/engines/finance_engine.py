"""Cashflow, spending behavior, and loan lifecycle updates."""

from __future__ import annotations

import math
import random
from typing import Dict, List, Optional

from financial_town.models import EmploymentStatus, Loan, LoanStatus, LoanType, Transaction, TransactionType
from financial_town.simulation.state import TownState

SPEND_CATEGORIES = [
  "groceries",
  "utilities",
  "transport",
  "healthcare",
  "education",
  "retail",
  "entertainment",
]

SPENDING_STYLE_RATIO = {
  "frugal": 0.10,
  "balanced": 0.18,
  "impulsive": 0.30,
}


class FinanceEngine:
  def __init__(self, rng: random.Random):
    self.rng = rng

  def _influence(self, state: TownState, person_id: str) -> Dict[str, float]:
    if person_id not in state.influence_state:
      state.influence_state[person_id] = {
        "spending_multiplier": 1.0,
        "loan_stress": 0.0,
        "fraud_susceptibility": 1.0,
      }
    return state.influence_state[person_id]

  def _next_tx_id(self, state: TownState) -> str:
    return f"T-{len(state.transactions) + 1:08d}"

  def _record_transaction(
    self,
    state: TownState,
    person_id: Optional[str],
    household_id: Optional[str],
    category: str,
    tx_type: TransactionType,
    amount: float,
    note: str = "",
    fraudulent: bool = False,
  ) -> None:
    state.transactions.append(
      Transaction(
        transaction_id=self._next_tx_id(state),
        step=state.step,
        timestamp=f"step-{state.step}",
        person_id=person_id,
        household_id=household_id,
        category=category,
        tx_type=tx_type,
        amount=round(max(0.0, amount), 2),
        fraudulent=fraudulent,
        note=note,
      )
    )

  def _withdraw(self, person, amount: float) -> float:
    """Withdraw from checking first then savings; returns unpaid amount."""
    remaining = amount
    if person.checking_balance > 0:
      used = min(person.checking_balance, remaining)
      person.checking_balance -= used
      remaining -= used
    if remaining > 0 and person.savings_balance > 0:
      used = min(person.savings_balance, remaining)
      person.savings_balance -= used
      remaining -= used
    person.checking_balance = round(person.checking_balance, 2)
    person.savings_balance = round(person.savings_balance, 2)
    return round(max(0.0, remaining), 2)

  def _originate_emergency_loan(self, state: TownState, person, shortfall: float) -> Loan:
    stress = max(0.0, self._influence(state, person.person_id).get("loan_stress", 0.0))
    principal = round(max(500.0, shortfall * (1.20 + min(0.80, 0.18 * stress))), 2)
    annual_rate = round(
      state.policy.central_bank_rate + state.policy.loan_rate_spread + self.rng.uniform(0.08, 0.16),
      4,
    )
    term = 24
    monthly_rate = annual_rate / 12.0
    monthly_payment = principal / term
    if monthly_rate > 0:
      monthly_payment = (monthly_rate * principal) / (1.0 - math.pow(1.0 + monthly_rate, -term))
    monthly_payment = round(monthly_payment, 2)

    loan_id = f"L-{len(state.loans) + 1:06d}"
    loan = Loan(
      loan_id=loan_id,
      borrower_person_id=person.person_id,
      loan_type=LoanType.EMERGENCY,
      purpose="liquidity_shortfall",
      principal=principal,
      remaining_principal=principal,
      annual_rate=annual_rate,
      remaining_term_months=term,
      monthly_payment=monthly_payment,
      status=LoanStatus.ACTIVE,
    )
    state.loans[loan_id] = loan
    person.loan_ids.append(loan_id)
    person.checking_balance = round(person.checking_balance + principal, 2)

    self._record_transaction(
      state=state,
      person_id=person.person_id,
      household_id=person.household_id,
      category="emergency_loan",
      tx_type=TransactionType.LOAN_ORIGINATION,
      amount=principal,
      note="auto-generated to cover monthly deficit",
    )
    influence = self._influence(state, person.person_id)
    influence["loan_stress"] = min(2.5, influence.get("loan_stress", 0.0) + 0.20)
    return loan

  def _gross_income(self, person, state: TownState) -> float:
    if person.employment_status == EmploymentStatus.EMPLOYED:
      gross = max(0.0, person.annual_base_income / 12.0)
      person.last_gross_income = round(gross, 2)
      return gross
    # Unemployment benefits anchored to previous gross pay.
    baseline = max(1200.0, person.last_gross_income)
    return baseline * state.policy.unemployment_replacement_rate

  def _monthly_spend_target(self, person, household, net_income: float, state: TownState) -> float:
    household_pressure = 0.34 + (0.06 * max(0, person.family_size - 1))
    style_ratio = SPENDING_STYLE_RATIO.get(person.spending_style, 0.18)
    # Inflation affects behavior modestly; explicit price indexes are applied at category level.
    macro_pressure = max(0.0, state.macro.inflation * 0.4)
    influence = self._influence(state, person.person_id)
    spending_multiplier = max(0.60, min(1.70, influence.get("spending_multiplier", 1.0)))
    loan_stress = max(0.0, min(2.5, influence.get("loan_stress", 0.0)))
    stress_pressure = 0.03 * loan_stress
    target_ratio = (household_pressure + style_ratio + macro_pressure + stress_pressure) * spending_multiplier
    return max(400.0, net_income * target_ratio)

  def _category_price_multiplier(self, state: TownState, category: str) -> float:
    cpi = 1.0
    try:
      cpi = float(getattr(state, "consumer_price_index", 1.0) or 1.0)
    except Exception:
      cpi = 1.0
    cpi = max(0.60, min(3.50, cpi))

    category_multipliers = getattr(state, "category_price_multipliers", {}) or {}
    cat = 1.0
    try:
      cat = float(category_multipliers.get(category, 1.0))
    except Exception:
      cat = 1.0
    cat = max(0.60, min(3.50, cat))
    return cpi * cat

  def _process_loan_payments(self, state: TownState, person) -> None:
    influence = self._influence(state, person.person_id)
    stress = max(0.0, min(2.8, influence.get("loan_stress", 0.0)))

    for loan_id in list(person.loan_ids):
      loan = state.loans.get(loan_id)
      if loan is None:
        continue
      if loan.status in {LoanStatus.CLOSED, LoanStatus.WRITTEN_OFF}:
        continue

      scheduled_payment = min(loan.monthly_payment, loan.remaining_principal)
      attempt_ratio = 1.0
      strategic_skip_prob = min(0.75, 0.10 + 0.25 * stress)
      if self.rng.random() < strategic_skip_prob:
        attempt_ratio = max(0.10, 1.0 - self.rng.uniform(0.18, 0.68))

      attempted_payment = round(scheduled_payment * attempt_ratio, 2)
      unpaid_attempt = self._withdraw(person, attempted_payment)
      paid = round(attempted_payment - unpaid_attempt, 2)
      unpaid_due = round(max(0.0, scheduled_payment - paid), 2)

      if paid > 0:
        interest_component = loan.remaining_principal * (loan.annual_rate / 12.0)
        principal_component = max(0.0, paid - interest_component)
        loan.remaining_principal = round(max(0.0, loan.remaining_principal - principal_component), 2)

      loan.remaining_term_months = max(0, loan.remaining_term_months - 1)
      if loan.remaining_principal <= 1.0 or loan.remaining_term_months == 0:
        loan.remaining_principal = 0.0
        loan.status = LoanStatus.CLOSED
        loan.delinquent_months = 0
        influence["loan_stress"] = max(0.0, influence.get("loan_stress", 0.0) - 0.18)
      elif unpaid_due > 0:
        loan.delinquent_months += 1
        loan.status = LoanStatus.DELINQUENT if loan.delinquent_months < 6 else LoanStatus.WRITTEN_OFF
        influence["loan_stress"] = min(2.8, influence.get("loan_stress", 0.0) + 0.24)
      else:
        loan.status = LoanStatus.ACTIVE
        loan.delinquent_months = 0
        influence["loan_stress"] = max(0.0, influence.get("loan_stress", 0.0) - 0.08)

      self._record_transaction(
        state=state,
        person_id=person.person_id,
        household_id=person.household_id,
        category=f"loan_payment:{loan.loan_type.value}",
        tx_type=TransactionType.LOAN_PAYMENT,
        amount=scheduled_payment,
        note="paid" if unpaid_due <= 0 else "partial_or_missed",
      )

  def _process_housing_cost(self, state: TownState, household) -> None:
    if not household.member_ids:
      return
    payer = state.people[household.member_ids[0]]
    due = max(0.0, household.monthly_housing_cost - state.policy.housing_subsidy)
    unpaid = self._withdraw(payer, due)
    if unpaid > 0:
      # Try one emergency loan, then retry.
      self._originate_emergency_loan(state, payer, unpaid)
      unpaid = self._withdraw(payer, unpaid)
      influence = self._influence(state, payer.person_id)
      if unpaid > 0:
        influence["loan_stress"] = min(2.8, influence.get("loan_stress", 0.0) + 0.20)
      else:
        influence["loan_stress"] = min(2.8, influence.get("loan_stress", 0.0) + 0.06)

    self._record_transaction(
      state=state,
      person_id=payer.person_id,
      household_id=household.household_id,
      category="housing_cost",
      tx_type=TransactionType.HOUSING,
      amount=due,
      note="paid" if unpaid <= 0 else "unpaid_tail",
    )

  def run_step(self, state: TownState) -> None:
    # 1) Income and day-to-day spending at person level.
    for person in state.people.values():
      gross_income = self._gross_income(person, state)
      payroll_tax = state.policy.payroll_tax_rate if person.employment_status == EmploymentStatus.EMPLOYED else 0.0
      taxes = gross_income * (state.policy.income_tax_rate + payroll_tax)
      net_income = max(0.0, gross_income - taxes + state.policy.stimulus_payment)
      person.checking_balance = round(person.checking_balance + net_income, 2)

      self._record_transaction(
        state=state,
        person_id=person.person_id,
        household_id=person.household_id,
        category="salary_or_benefit",
        tx_type=TransactionType.INCOME,
        amount=net_income,
      )

      household = state.households[person.household_id]
      spend_target = self._monthly_spend_target(person, household, net_income, state)

      # Split spend across categories.
      raw_weights = [self.rng.uniform(0.2, 1.4) for _ in SPEND_CATEGORIES]
      weight_sum = sum(raw_weights)
      for category, weight in zip(SPEND_CATEGORIES, raw_weights):
        base_amount = spend_target * (weight / weight_sum)
        price_multiplier = self._category_price_multiplier(state, category)
        amount = base_amount * price_multiplier
        unpaid = self._withdraw(person, amount)
        if unpaid > 0:
          self._originate_emergency_loan(state, person, unpaid)
          self._withdraw(person, unpaid)
        self._record_transaction(
          state=state,
          person_id=person.person_id,
          household_id=person.household_id,
          category=category,
          tx_type=TransactionType.EXPENSE,
          amount=amount,
          note=f"base={base_amount:.2f};price_mult={price_multiplier:.3f}",
        )

    # 2) Housing costs once per household.
    for household in state.households.values():
      self._process_housing_cost(state, household)

    # 3) Loan payment lifecycle.
    for person in state.people.values():
      self._process_loan_payments(state, person)

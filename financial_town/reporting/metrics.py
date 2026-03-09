"""Step-level metrics for monitoring town dynamics."""

from __future__ import annotations

import statistics
from typing import Iterable

from financial_town.models import EmploymentStatus, LoanStatus
from financial_town.simulation.state import TownState


def _gini(values: Iterable[float]) -> float:
  arr = sorted(max(0.0, v) for v in values)
  if not arr:
    return 0.0
  total = sum(arr)
  if total <= 0:
    return 0.0
  n = len(arr)
  weighted_sum = sum((idx + 1) * value for idx, value in enumerate(arr))
  return (2 * weighted_sum) / (n * total) - (n + 1) / n


class MetricsEngine:
  def compute_step_metrics(self, state: TownState) -> dict:
    people = list(state.people.values())
    loans = list(state.loans.values())
    properties = list(state.properties.values())

    employed = [p for p in people if p.employment_status == EmploymentStatus.EMPLOYED]
    annual_income = [p.annual_base_income for p in people]
    liquid_cash = [p.checking_balance + p.savings_balance for p in people]

    total_property_value = sum(p.market_value for p in properties)
    total_debt = sum(l.remaining_principal for l in loans if l.status != LoanStatus.CLOSED)
    active_loans = [l for l in loans if l.status in {LoanStatus.ACTIVE, LoanStatus.DELINQUENT}]
    delinquent_loans = [l for l in loans if l.status == LoanStatus.DELINQUENT]

    fraud_count_step = len([tx for tx in state.transactions if tx.step == state.step and tx.fraudulent])
    fraud_count_total = len([tx for tx in state.transactions if tx.fraudulent])
    interactions_step = len([evt for evt in state.interaction_events if evt.get("step") == state.step])
    avg_spending_multiplier = statistics.mean(
      [mods.get("spending_multiplier", 1.0) for mods in state.influence_state.values()]
    ) if state.influence_state else 1.0
    avg_loan_stress = statistics.mean(
      [mods.get("loan_stress", 0.0) for mods in state.influence_state.values()]
    ) if state.influence_state else 0.0
    avg_fraud_susceptibility = statistics.mean(
      [mods.get("fraud_susceptibility", 1.0) for mods in state.influence_state.values()]
    ) if state.influence_state else 1.0
    category_prices = list((getattr(state, "category_price_multipliers", {}) or {}).values())
    avg_category_price_multiplier = statistics.mean(category_prices) if category_prices else 1.0

    llm_calls_step = int(state.llm_stats.get("calls_step", 0.0)) if state.llm_stats else 0
    llm_calls_total = int(state.llm_stats.get("calls_total", 0.0)) if state.llm_stats else 0
    llm_errors_step = int(state.llm_stats.get("errors_step", 0.0)) if state.llm_stats else 0
    llm_errors_total = int(state.llm_stats.get("errors_total", 0.0)) if state.llm_stats else 0
    llm_prompt_tokens_step = int(state.llm_stats.get("prompt_tokens_step", 0.0)) if state.llm_stats else 0
    llm_completion_tokens_step = int(state.llm_stats.get("completion_tokens_step", 0.0)) if state.llm_stats else 0
    llm_prompt_tokens_total = int(state.llm_stats.get("prompt_tokens_total", 0.0)) if state.llm_stats else 0
    llm_completion_tokens_total = int(state.llm_stats.get("completion_tokens_total", 0.0)) if state.llm_stats else 0

    return {
      "step": state.step,
      "scenario": state.scenario_name,
      "population": len(people),
      "employed_rate": round(len(employed) / max(1, len(people)), 4),
      "avg_annual_income": round(sum(annual_income) / max(1, len(people)), 2),
      "median_liquid_cash": round(statistics.median(liquid_cash) if liquid_cash else 0.0, 2),
      "gini_income": round(_gini(annual_income), 4),
      "total_property_value": round(total_property_value, 2),
      "total_debt": round(total_debt, 2),
      "loan_delinquency_rate": round(len(delinquent_loans) / max(1, len(active_loans)), 4),
      "fraudulent_txn_step": fraud_count_step,
      "fraudulent_txn_total": fraud_count_total,
      "interaction_events_step": interactions_step,
      "interaction_events_total": len(state.interaction_events),
      "avg_spending_multiplier": round(avg_spending_multiplier, 4),
      "avg_loan_stress": round(avg_loan_stress, 4),
      "avg_fraud_susceptibility": round(avg_fraud_susceptibility, 4),
      "llm_calls_step": llm_calls_step,
      "llm_calls_total": llm_calls_total,
      "llm_errors_step": llm_errors_step,
      "llm_errors_total": llm_errors_total,
      "llm_prompt_tokens_step": llm_prompt_tokens_step,
      "llm_completion_tokens_step": llm_completion_tokens_step,
      "llm_prompt_tokens_total": llm_prompt_tokens_total,
      "llm_completion_tokens_total": llm_completion_tokens_total,
      "macro_events_total": len(state.macro_events),
      "active_agent_memory_size": len(state.active_agent_memory),
      "consumer_price_index": round(float(getattr(state, "consumer_price_index", 1.0) or 1.0), 4),
      "avg_category_price_multiplier": round(float(avg_category_price_multiplier), 4),
      "macro_gdp_growth": round(state.macro.gdp_growth, 4),
      "macro_inflation": round(state.macro.inflation, 4),
      "macro_unemployment": round(state.macro.unemployment_rate, 4),
    }

"""Fraud and anomaly injection for model training / stress tests."""

from __future__ import annotations

import random
from typing import List

from financial_town.config import SimulationConfig
from financial_town.models import Transaction, TransactionType
from financial_town.simulation.state import TownState


class FraudEngine:
  def __init__(self, rng: random.Random, config: SimulationConfig):
    self.rng = rng
    self.config = config

  def _fraud_susceptibility(self, state: TownState, person_id: str) -> float:
    mods = state.influence_state.get(person_id, {})
    base = float(mods.get("fraud_susceptibility", 1.0))
    return max(0.45, min(2.4, base))

  def _candidate_transactions(self, state: TownState) -> List[Transaction]:
    return [
      tx for tx in state.transactions
      if tx.step == state.step
      and tx.tx_type in {TransactionType.EXPENSE, TransactionType.HOUSING, TransactionType.LOAN_PAYMENT}
    ]

  def _inject_transaction_anomalies(self, state: TownState) -> int:
    detect_modifier = max(0.0, 1.0 - state.policy.fraud_detection_strength)
    anomaly_prob = self.config.fraud_transaction_rate * (0.45 + detect_modifier)
    count = 0
    for tx in self._candidate_transactions(state):
      susceptibility = 1.0
      if tx.person_id:
        susceptibility = self._fraud_susceptibility(state, tx.person_id)
      tx_prob = anomaly_prob * (0.70 + 0.35 * susceptibility)
      if self.rng.random() < tx_prob:
        tx.fraudulent = True
        tx.amount = round(tx.amount * self.rng.uniform(2.2, 7.0), 2)
        tx.note = "synthetic_fraudulent_spike"
        count += 1
        if tx.person_id:
          mods = state.influence_state.setdefault(tx.person_id, {})
          mods["fraud_susceptibility"] = min(2.5, float(mods.get("fraud_susceptibility", 1.0)) + 0.06)
    return count

  def _spread_fraud_risk(self, state: TownState, source_person_id: str) -> None:
    neighbors = state.social_graph.get(source_person_id, {})
    for neighbor_id, relations in neighbors.items():
      relation_boost = 0.02
      if "family" in relations:
        relation_boost += 0.06
      if "coworker" in relations:
        relation_boost += 0.04
      if "neighbor" in relations:
        relation_boost += 0.03
      if self.rng.random() < min(0.88, 0.20 + relation_boost):
        mods = state.influence_state.setdefault(neighbor_id, {})
        mods["fraud_susceptibility"] = min(
          2.6,
          float(mods.get("fraud_susceptibility", 1.0)) + relation_boost,
        )

  def _inject_scam_lending_events(self, state: TownState) -> int:
    base_attempts = int(len(state.people) * self.config.fraudulent_loan_rate)
    if self.rng.random() < (len(state.people) * self.config.fraudulent_loan_rate - base_attempts):
      base_attempts += 1

    people = list(state.people.values())
    if not people:
      return 0

    weights = [self._fraud_susceptibility(state, p.person_id) for p in people]
    generated = 0
    for _ in range(base_attempts):
      person = self.rng.choices(people, weights=weights, k=1)[0]
      tx = Transaction(
        transaction_id=f"T-{len(state.transactions) + 1:08d}",
        step=state.step,
        timestamp=f"step-{state.step}",
        person_id=person.person_id,
        household_id=person.household_id,
        category="synthetic_scam_lending",
        tx_type=TransactionType.FRAUD_ALERT,
        amount=round(self.rng.uniform(3500, 125000), 2),
        fraudulent=True,
        note="synthetic_false_application_or_bust_out",
      )
      state.transactions.append(tx)
      self._spread_fraud_risk(state, person.person_id)
      generated += 1
    return generated

  def run_step(self, state: TownState) -> None:
    self._inject_transaction_anomalies(state)
    self._inject_scam_lending_events(state)

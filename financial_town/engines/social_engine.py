"""Social interaction and influence diffusion across network edges."""

from __future__ import annotations

import random
from typing import List

from financial_town.models import LoanStatus
from financial_town.simulation.state import TownState


class SocialEngine:
  def __init__(self, rng: random.Random):
    self.rng = rng

  def _ensure_influence(self, state: TownState, person_id: str):
    if person_id not in state.influence_state:
      state.influence_state[person_id] = {
        "spending_multiplier": 1.0,
        "loan_stress": 0.0,
        "fraud_susceptibility": 1.0,
      }
    return state.influence_state[person_id]

  def _decay_influence(self, state: TownState) -> None:
    for person_id in state.people:
      mods = self._ensure_influence(state, person_id)
      mods["spending_multiplier"] = 1.0 + (mods["spending_multiplier"] - 1.0) * 0.75
      mods["loan_stress"] = max(0.0, mods["loan_stress"] * 0.72)
      mods["fraud_susceptibility"] = 1.0 + (mods["fraud_susceptibility"] - 1.0) * 0.78

  def _interaction_probability(self, relations: List[str], distance: int, same_place: bool) -> float:
    base = 0.05
    if "family" in relations:
      base += 0.26
    if "coworker" in relations:
      base += 0.18
    if "neighbor" in relations:
      base += 0.12
    if same_place:
      base += 0.14
    if distance <= 2:
      base += 0.10
    elif distance <= 5:
      base += 0.03
    else:
      base -= 0.06
    return max(0.02, min(0.92, base))

  def _topic_for_interaction(self, relations: List[str], same_place: bool) -> str:
    topics = [
      ("budget_planning", 0.25),
      ("spending_urge", 0.25),
      ("loan_stress", 0.20),
      ("fraud_warning", 0.15),
      ("scam_rumor", 0.15),
    ]
    if "coworker" in relations and same_place:
      topics = [
        ("budget_planning", 0.18),
        ("spending_urge", 0.20),
        ("loan_stress", 0.22),
        ("fraud_warning", 0.23),
        ("scam_rumor", 0.17),
      ]
    if "family" in relations:
      topics = [
        ("budget_planning", 0.28),
        ("spending_urge", 0.16),
        ("loan_stress", 0.28),
        ("fraud_warning", 0.20),
        ("scam_rumor", 0.08),
      ]
    labels = [x[0] for x in topics]
    probs = [x[1] for x in topics]
    return self.rng.choices(labels, weights=probs, k=1)[0]

  def _apply_topic_effect(self, state: TownState, source_id: str, target_id: str, topic: str) -> None:
    target = self._ensure_influence(state, target_id)
    if topic == "budget_planning":
      target["spending_multiplier"] = max(0.65, target["spending_multiplier"] - 0.08)
    elif topic == "spending_urge":
      target["spending_multiplier"] = min(1.55, target["spending_multiplier"] + 0.14)
    elif topic == "loan_stress":
      target["loan_stress"] = min(2.0, target["loan_stress"] + 0.22)
    elif topic == "fraud_warning":
      target["fraud_susceptibility"] = max(0.55, target["fraud_susceptibility"] - 0.14)
    elif topic == "scam_rumor":
      target["fraud_susceptibility"] = min(1.8, target["fraud_susceptibility"] + 0.18)

    # Delinquency contagion: if source is delinquent, target stress rises.
    source = state.people[source_id]
    source_has_delinquency = False
    for loan_id in source.loan_ids:
      loan = state.loans.get(loan_id)
      if loan and loan.status == LoanStatus.DELINQUENT:
        source_has_delinquency = True
        break
    if source_has_delinquency:
      target["loan_stress"] = min(2.2, target["loan_stress"] + 0.12)

  def _chat_text(self, p1_name: str, p2_name: str, topic: str) -> str:
    mapping = {
      "budget_planning": f"{p1_name} and {p2_name} discussed budget planning.",
      "spending_urge": f"{p1_name} and {p2_name} talked about buying new things.",
      "loan_stress": f"{p1_name} and {p2_name} discussed loan stress.",
      "fraud_warning": f"{p1_name} warned {p2_name} about suspicious lending scams.",
      "scam_rumor": f"{p1_name} and {p2_name} exchanged a risky lending rumor.",
    }
    return mapping.get(topic, f"{p1_name} and {p2_name} had a quick chat.")

  def run_step(self, state: TownState) -> None:
    self._decay_influence(state)

    processed_pairs = set()
    for p1_id, neighbors in state.social_graph.items():
      for p2_id, relations in neighbors.items():
        pair_key = tuple(sorted([p1_id, p2_id]))
        if pair_key in processed_pairs:
          continue
        processed_pairs.add(pair_key)
        if p1_id not in state.people or p2_id not in state.people:
          continue

        p1 = state.people[p1_id]
        p2 = state.people[p2_id]
        distance = abs(p1.current_x - p2.current_x) + abs(p1.current_y - p2.current_y)
        same_place = p1.current_place == p2.current_place
        prob = self._interaction_probability(relations, distance, same_place)

        if self.rng.random() < prob:
          topic = self._topic_for_interaction(relations, same_place)
          if self.rng.random() < 0.5:
            source_id, target_id = p1_id, p2_id
          else:
            source_id, target_id = p2_id, p1_id
          self._apply_topic_effect(state, source_id, target_id, topic)

          chat_text = self._chat_text(p1.full_name, p2.full_name, topic)
          p1_chat = [[p2.full_name, chat_text]]
          p2_chat = [[p1.full_name, chat_text]]
          state.step_chat[p1_id] = p1_chat
          state.step_chat[p2_id] = p2_chat
          if p1_id in state.step_movements:
            state.step_movements[p1_id]["chat"] = p1_chat
          if p2_id in state.step_movements:
            state.step_movements[p2_id]["chat"] = p2_chat

          state.interaction_events.append({
            "step": state.step,
            "source_id": source_id,
            "target_id": target_id,
            "people": [p1.full_name, p2.full_name],
            "relations": relations,
            "topic": topic,
            "distance": distance,
            "same_place": same_place,
          })

"""Mutable simulation state container."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

from financial_town.models import Household, Loan, MacroState, Person, PolicyState, Property, Transaction


@dataclass
class TownState:
  step: int
  scenario_name: str

  people: Dict[str, Person]
  households: Dict[str, Household]
  properties: Dict[str, Property]
  loans: Dict[str, Loan]

  policy: PolicyState
  macro: MacroState

  transactions: List[Transaction] = field(default_factory=list)
  metrics_history: List[dict] = field(default_factory=list)
  social_graph: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)
  influence_state: Dict[str, Dict[str, float]] = field(default_factory=dict)
  interaction_events: List[dict] = field(default_factory=list)
  step_chat: Dict[str, list] = field(default_factory=dict)
  step_movements: Dict[str, dict] = field(default_factory=dict)
  llm_events: List[dict] = field(default_factory=list)
  llm_stats: Dict[str, float] = field(default_factory=dict)
  active_agent_id: str = ""
  scripted_agent_ids: List[str] = field(default_factory=list)
  active_agent_memory: List[dict] = field(default_factory=list)
  active_agent_thoughts: List[dict] = field(default_factory=list)
  macro_events: List[dict] = field(default_factory=list)
  active_agent_runtime_context: Dict[str, object] = field(default_factory=dict)
  active_agent_plan: Dict[str, object] = field(default_factory=dict)
  active_agent_mood: Dict[str, float] = field(default_factory=lambda: {
    "valence": 0.6,
    "energy": 0.7,
    "stress": 0.2,
  })
  active_agent_reflections: List[dict] = field(default_factory=list)
  active_agent_relationships: Dict[str, Dict[str, object]] = field(default_factory=dict)
  # Price system used by finance engine and surfaced to the active agent.
  consumer_price_index: float = 1.0
  category_price_multipliers: Dict[str, float] = field(default_factory=dict)

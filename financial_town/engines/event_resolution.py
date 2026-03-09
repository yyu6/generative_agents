"""Event resolution pipeline: validate, resolve, and propagate macro events.

Inspired by Concordia's EventResolution pattern (game master as referee).
The pipeline ensures LLM-generated events are:
  1. Structurally valid (correct keys, types, ranges)
  2. Economically consistent (no contradictions like deflation + rate hikes)
  3. Resolved through a GM-style adjudication step
  4. Propagated through deterministic chain reactions
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from financial_town.config import SimulationConfig
from financial_town.llm import AgentLLMClient
from financial_town.simulation.state import TownState

SPEND_CATEGORIES = [
  "groceries", "utilities", "transport", "healthcare",
  "education", "retail", "entertainment",
]

MACRO_FIELD_BOUNDS = {
  "gdp_growth":        (-0.08, 0.12),
  "inflation":         (-0.01, 0.20),
  "unemployment_rate": (0.01, 0.35),
  "housing_growth":    (-0.12, 0.12),
}

POLICY_FIELD_BOUNDS = {
  "layoff_rate":       (0.001, 0.25),
  "stimulus_payment":  (0.0, 1000.0),
  "housing_subsidy":   (0.0, 1200.0),
  "central_bank_rate": (0.0, 0.25),
}

MAX_DELTA_PER_STEP = {
  "gdp_growth": 0.015,
  "inflation": 0.02,
  "unemployment_rate": 0.03,
  "housing_growth": 0.02,
  "layoff_rate": 0.03,
  "stimulus_payment": 200.0,
  "housing_subsidy": 200.0,
  "central_bank_rate": 0.015,
  "global_price_pct": 0.08,
  "category_price_pct": 0.12,
}

CONSISTENCY_RULES = [
  {
    "name": "deflation_with_rate_hike",
    "description": "Central bank should not raise rates during deflation",
    "check": lambda md, pd: not (
      md.get("inflation", 0) < -0.005
      and pd.get("central_bank_rate", 0) > 0.002
    ),
  },
  {
    "name": "stimulus_during_boom",
    "description": "Stimulus should not increase during strong GDP growth",
    "check": lambda md, pd: not (
      md.get("gdp_growth", 0) > 0.01
      and pd.get("stimulus_payment", 0) > 50
    ),
  },
  {
    "name": "unemployment_drop_with_layoffs",
    "description": "Unemployment should not drop while layoff rate rises sharply",
    "check": lambda md, pd: not (
      md.get("unemployment_rate", 0) < -0.01
      and pd.get("layoff_rate", 0) > 0.01
    ),
  },
  {
    "name": "gdp_unemployment_consistency",
    "description": "GDP and unemployment should not both move in the same positive direction",
    "check": lambda md, pd: not (
      md.get("gdp_growth", 0) > 0.005
      and md.get("unemployment_rate", 0) > 0.005
    ),
  },
]


@dataclass
class ValidationResult:
  valid: bool
  violations: List[str] = field(default_factory=list)
  clamped_macro_delta: Dict[str, float] = field(default_factory=dict)
  clamped_policy_delta: Dict[str, float] = field(default_factory=dict)
  clamped_price_shock: Dict[str, object] = field(default_factory=dict)


@dataclass
class ResolvedEvent:
  note: str
  macro_delta: Dict[str, float]
  policy_delta: Dict[str, float]
  price_shock: Dict[str, object]
  chain_reactions: List[Dict[str, object]]
  resolution_reasoning: str
  validation: ValidationResult
  rejected: bool = False
  rejection_reason: str = ""


class EventValidator:
  """Hard constraint validation for LLM-generated economic events."""

  def validate(
    self,
    macro_delta: Dict[str, float],
    policy_delta: Dict[str, float],
    price_shock: Dict[str, object],
    current_state: TownState,
  ) -> ValidationResult:
    violations: List[str] = []
    clamped_macro = {}
    clamped_policy = {}
    clamped_price: Dict[str, object] = {}

    for field_name, delta_val in macro_delta.items():
      delta = self._safe_float(delta_val)
      max_d = MAX_DELTA_PER_STEP.get(field_name, 0.05)
      clamped = max(-max_d, min(max_d, delta))
      if abs(delta) > max_d:
        violations.append(
          f"macro.{field_name} delta {delta:.4f} clamped to {clamped:.4f} "
          f"(max per-step: {max_d})"
        )
      bounds = MACRO_FIELD_BOUNDS.get(field_name)
      if bounds:
        current = getattr(current_state.macro, field_name, 0.0)
        new_val = max(bounds[0], min(bounds[1], current + clamped))
        clamped = new_val - current
      clamped_macro[field_name] = round(clamped, 6)

    for field_name, delta_val in policy_delta.items():
      delta = self._safe_float(delta_val)
      max_d = MAX_DELTA_PER_STEP.get(field_name, 0.05)
      clamped = max(-max_d, min(max_d, delta))
      if abs(delta) > max_d:
        violations.append(
          f"policy.{field_name} delta {delta:.4f} clamped to {clamped:.4f}"
        )
      bounds = POLICY_FIELD_BOUNDS.get(field_name)
      if bounds:
        current = getattr(current_state.policy, field_name, 0.0)
        new_val = max(bounds[0], min(bounds[1], current + clamped))
        clamped = new_val - current
      clamped_policy[field_name] = round(clamped, 6)

    global_pct = self._safe_float(
      price_shock.get("global_price_pct", 0.0)
    )
    max_g = MAX_DELTA_PER_STEP["global_price_pct"]
    clamped_global = max(-max_g, min(max_g, global_pct))
    if abs(global_pct) > max_g:
      violations.append(
        f"global_price_pct {global_pct:.4f} clamped to {clamped_global:.4f}"
      )

    cat_raw = price_shock.get("category_price_pct", {})
    if not isinstance(cat_raw, dict):
      cat_raw = {}
    clamped_cat = {}
    max_c = MAX_DELTA_PER_STEP["category_price_pct"]
    for cat in SPEND_CATEGORIES:
      val = self._safe_float(cat_raw.get(cat, 0.0))
      clamped_cat[cat] = round(max(-max_c, min(max_c, val)), 6)

    clamped_price = {
      "global_price_pct": round(clamped_global, 6),
      "category_price_pct": clamped_cat,
    }

    for rule in CONSISTENCY_RULES:
      if not rule["check"](clamped_macro, clamped_policy):
        violations.append(f"consistency: {rule['name']} -- {rule['description']}")

    valid = len([v for v in violations if "consistency:" in v]) == 0

    return ValidationResult(
      valid=valid,
      violations=violations,
      clamped_macro_delta=clamped_macro,
      clamped_policy_delta=clamped_policy,
      clamped_price_shock=clamped_price,
    )

  def _safe_float(self, value, default: float = 0.0) -> float:
    try:
      return float(value)
    except Exception:
      return default


class EventResolver:
  """Concordia-style Game Master that adjudicates putative events.

  Takes a proposed event and the current world state, asks the LLM to
  assess plausibility and determine what actually happens. This acts
  as a "referee" that can modify, reject, or accept events.
  """

  def __init__(self, config: SimulationConfig, rng: random.Random):
    self.config = config
    self.rng = rng
    self.client = AgentLLMClient(
      provider=config.llm_economist_provider,
      model=config.llm_economist_model,
      temperature=max(0.0, config.llm_economist_temperature - 0.1),
      max_output_tokens=config.llm_economist_max_output_tokens + 120,
      timeout_sec=config.llm_timeout_sec,
    )

  def _system_prompt(self) -> str:
    return (
      "You are a macroeconomic referee (Game Master) in an agent-based "
      "economy simulation. Your job is to evaluate whether a proposed "
      "economic event is PLAUSIBLE given the current world state.\n\n"
      "You must check:\n"
      "1. ECONOMIC CAUSALITY: Does the event logically follow from current "
      "conditions? (e.g. rate cuts during high inflation are unusual)\n"
      "2. MAGNITUDE REASONABLENESS: Are the changes realistic for a single "
      "period? (GDP doesn't jump 5% in one step)\n"
      "3. INTERNAL CONSISTENCY: Do the macro/policy/price changes make "
      "sense together? (e.g. deflation + price increases is contradictory)\n"
      "4. HISTORICAL PRECEDENT: Could this happen in a real economy?\n\n"
      "Return JSON with keys:\n"
      "- accepted: boolean (true if event is plausible, false if rejected)\n"
      "- reasoning: string (1-2 sentences explaining your judgment)\n"
      "- adjusted_note: string (revised event description if accepted, "
      "or rejection reason if not)\n"
      "- adjustments: object with optional macro_delta, policy_delta, "
      "price_shock overrides if you think the magnitudes need correction. "
      "Only include fields you want to change. Omit if the original is fine."
    )

  def _user_prompt(self, state: TownState, proposed_event: Dict[str, object]) -> str:
    payload = {
      "current_world_state": {
        "macro": {
          "gdp_growth": state.macro.gdp_growth,
          "inflation": state.macro.inflation,
          "unemployment_rate": state.macro.unemployment_rate,
          "housing_growth": state.macro.housing_growth,
        },
        "policy": {
          "layoff_rate": state.policy.layoff_rate,
          "stimulus_payment": state.policy.stimulus_payment,
          "housing_subsidy": state.policy.housing_subsidy,
          "central_bank_rate": state.policy.central_bank_rate,
        },
        "prices": {
          "consumer_price_index": round(
            float(getattr(state, "consumer_price_index", 1.0) or 1.0), 4
          ),
        },
        "population": len(state.people),
        "employed": len([
          p for p in state.people.values()
          if p.employment_status.value == "employed"
        ]),
        "total_debt": round(
          sum(l.remaining_principal for l in state.loans.values()), 2
        ),
      },
      "recent_events": [
        {"step": e.get("step"), "note": e.get("note", "")}
        for e in state.macro_events[-4:]
      ],
      "proposed_event": proposed_event,
    }
    return (
      "Evaluate this proposed economic event. "
      "Is it plausible given the current world state?\n"
      + json.dumps(payload, indent=2)
    )

  def resolve(
    self,
    state: TownState,
    proposed_event: Dict[str, object],
  ) -> Tuple[bool, str, Dict[str, object]]:
    result = self.client.generate_json(
      self._system_prompt(),
      self._user_prompt(state, proposed_event),
    )

    prompt_tokens = int(result.get("prompt_tokens", 0))
    completion_tokens = int(result.get("completion_tokens", 0))

    if not result.get("ok"):
      return True, "Resolution LLM failed; accepting with original values", {}

    payload = result.get("payload", {})
    accepted = bool(payload.get("accepted", True))
    reasoning = str(payload.get("reasoning", "")).strip()[:300]
    adjusted_note = str(payload.get("adjusted_note", "")).strip()[:260]
    adjustments = payload.get("adjustments", {})
    if not isinstance(adjustments, dict):
      adjustments = {}

    return (
      accepted,
      reasoning or adjusted_note or "no reasoning provided",
      adjustments,
    )


class ChainReactionPipeline:
  """Computes deterministic cascading effects from a validated event.

  After a macro event is accepted, this computes secondary effects using
  fixed economic rules (no LLM needed for most chains).
  """

  def __init__(self, rng: random.Random):
    self.rng = rng

  def compute_chains(
    self,
    macro_delta: Dict[str, float],
    policy_delta: Dict[str, float],
    price_shock: Dict[str, object],
    state: TownState,
  ) -> List[Dict[str, object]]:
    chains: List[Dict[str, object]] = []

    inflation_d = macro_delta.get("inflation", 0.0)
    gdp_d = macro_delta.get("gdp_growth", 0.0)
    unemp_d = macro_delta.get("unemployment_rate", 0.0)
    rate_d = policy_delta.get("central_bank_rate", 0.0)
    current_inflation = state.macro.inflation
    current_rate = state.policy.central_bank_rate

    if inflation_d > 0.008:
      rate_response = round(inflation_d * 0.4, 5)
      chains.append({
        "type": "monetary_policy_response",
        "trigger": f"inflation rose by {inflation_d:.3f}",
        "effect": "central_bank_rate",
        "delta": rate_response,
        "reasoning": "Central bank responds to rising inflation with rate increase",
      })

    if inflation_d < -0.005 and current_inflation < 0.01:
      rate_cut = round(abs(inflation_d) * 0.3, 5)
      chains.append({
        "type": "monetary_policy_response",
        "trigger": f"inflation fell by {inflation_d:.3f}, risk of deflation",
        "effect": "central_bank_rate",
        "delta": -rate_cut,
        "reasoning": "Central bank cuts rates to prevent deflation",
      })

    if unemp_d > 0.015:
      stim = round(unemp_d * 2000, 2)
      chains.append({
        "type": "fiscal_stimulus_response",
        "trigger": f"unemployment rose sharply by {unemp_d:.3f}",
        "effect": "stimulus_payment",
        "delta": min(stim, 150.0),
        "reasoning": "Government increases stimulus in response to rising unemployment",
      })

    if gdp_d < -0.01:
      unemp_effect = round(abs(gdp_d) * 0.6, 5)
      chains.append({
        "type": "labor_market_contraction",
        "trigger": f"GDP contracted by {gdp_d:.3f}",
        "effect": "unemployment_rate",
        "delta": unemp_effect,
        "reasoning": "GDP contraction leads to higher unemployment",
      })

    if gdp_d > 0.008:
      unemp_relief = round(gdp_d * 0.3, 5)
      chains.append({
        "type": "labor_market_expansion",
        "trigger": f"GDP grew by {gdp_d:.3f}",
        "effect": "unemployment_rate",
        "delta": -unemp_relief,
        "reasoning": "GDP growth creates jobs, reducing unemployment",
      })

    if rate_d > 0.005:
      housing_impact = round(rate_d * -1.5, 5)
      chains.append({
        "type": "housing_rate_sensitivity",
        "trigger": f"central bank rate increased by {rate_d:.4f}",
        "effect": "housing_growth",
        "delta": housing_impact,
        "reasoning": "Higher interest rates slow housing market growth",
      })

    global_price = float(
      price_shock.get("global_price_pct", 0.0) if isinstance(price_shock, dict) else 0.0
    )
    if abs(global_price) > 0.03:
      inflation_from_prices = round(global_price * 0.4, 5)
      chains.append({
        "type": "price_inflation_pass_through",
        "trigger": f"global prices shifted by {global_price:.3f}",
        "effect": "inflation",
        "delta": inflation_from_prices,
        "reasoning": "Price shocks feed into measured inflation",
      })

    return chains

  def apply_chains(
    self,
    chains: List[Dict[str, object]],
    state: TownState,
  ) -> None:
    for chain in chains:
      effect = chain.get("effect", "")
      delta = float(chain.get("delta", 0.0))

      if effect in MACRO_FIELD_BOUNDS:
        bounds = MACRO_FIELD_BOUNDS[effect]
        current = getattr(state.macro, effect, 0.0)
        max_chain_d = MAX_DELTA_PER_STEP.get(effect, 0.05) * 0.5
        clamped_delta = max(-max_chain_d, min(max_chain_d, delta))
        new_val = max(bounds[0], min(bounds[1], current + clamped_delta))
        setattr(state.macro, effect, new_val)
        chain["applied_delta"] = round(new_val - current, 6)

      elif effect in POLICY_FIELD_BOUNDS:
        bounds = POLICY_FIELD_BOUNDS[effect]
        current = getattr(state.policy, effect, 0.0)
        max_chain_d = MAX_DELTA_PER_STEP.get(effect, 0.05) * 0.5
        clamped_delta = max(-max_chain_d, min(max_chain_d, delta))
        new_val = max(bounds[0], min(bounds[1], current + clamped_delta))
        setattr(state.policy, effect, new_val)
        chain["applied_delta"] = round(new_val - current, 6)


class EventResolutionPipeline:
  """Full pipeline: generate -> validate -> resolve -> chain react.

  Replaces direct event application with a multi-stage verified process.
  """

  def __init__(self, config: SimulationConfig, rng: random.Random):
    self.config = config
    self.rng = rng
    self.validator = EventValidator()
    self.resolver = EventResolver(config, rng)
    self.chain_pipeline = ChainReactionPipeline(rng)
    self._rejection_count = 0
    self._total_count = 0

  def process_event(
    self,
    state: TownState,
    raw_event: Dict[str, object],
  ) -> ResolvedEvent:
    self._total_count += 1
    note = str(raw_event.get("note", "LLM-Economist event")).strip()[:260]
    macro_delta = raw_event.get("macro_delta", {})
    policy_delta = raw_event.get("policy_delta", {})
    price_shock = raw_event.get("price_shock", {})
    if not isinstance(macro_delta, dict):
      macro_delta = {}
    if not isinstance(policy_delta, dict):
      policy_delta = {}
    if not isinstance(price_shock, dict):
      price_shock = {}

    validation = self.validator.validate(
      macro_delta, policy_delta, price_shock, state,
    )

    if not validation.valid:
      self._rejection_count += 1
      return ResolvedEvent(
        note=note,
        macro_delta=validation.clamped_macro_delta,
        policy_delta=validation.clamped_policy_delta,
        price_shock=validation.clamped_price_shock,
        chain_reactions=[],
        resolution_reasoning="Rejected by structural validator: " + "; ".join(validation.violations),
        validation=validation,
        rejected=True,
        rejection_reason="; ".join(validation.violations),
      )

    proposed_for_resolution = {
      "note": note,
      "macro_delta": validation.clamped_macro_delta,
      "policy_delta": validation.clamped_policy_delta,
      "price_shock": validation.clamped_price_shock,
    }

    accepted, reasoning, adjustments = self.resolver.resolve(
      state, proposed_for_resolution,
    )

    if not accepted:
      self._rejection_count += 1
      return ResolvedEvent(
        note=note,
        macro_delta=validation.clamped_macro_delta,
        policy_delta=validation.clamped_policy_delta,
        price_shock=validation.clamped_price_shock,
        chain_reactions=[],
        resolution_reasoning=reasoning,
        validation=validation,
        rejected=True,
        rejection_reason=reasoning,
      )

    final_macro = dict(validation.clamped_macro_delta)
    final_policy = dict(validation.clamped_policy_delta)
    final_price = dict(validation.clamped_price_shock)

    if adjustments:
      adj_macro = adjustments.get("macro_delta", {})
      adj_policy = adjustments.get("policy_delta", {})
      adj_price = adjustments.get("price_shock", {})
      if isinstance(adj_macro, dict):
        for k, v in adj_macro.items():
          try:
            final_macro[k] = float(v)
          except Exception:
            pass
      if isinstance(adj_policy, dict):
        for k, v in adj_policy.items():
          try:
            final_policy[k] = float(v)
          except Exception:
            pass
      if isinstance(adj_price, dict):
        for k, v in adj_price.items():
          final_price[k] = v

      re_validation = self.validator.validate(
        final_macro, final_policy, final_price, state,
      )
      final_macro = re_validation.clamped_macro_delta
      final_policy = re_validation.clamped_policy_delta
      final_price = re_validation.clamped_price_shock

    adjusted_note = str(adjustments.get("adjusted_note", note)).strip()[:260]
    if not adjusted_note:
      adjusted_note = note

    chains = self.chain_pipeline.compute_chains(
      final_macro, final_policy, final_price, state,
    )

    self._apply_to_state(state, final_macro, final_policy, final_price)
    self.chain_pipeline.apply_chains(chains, state)

    return ResolvedEvent(
      note=adjusted_note,
      macro_delta=final_macro,
      policy_delta=final_policy,
      price_shock=final_price,
      chain_reactions=chains,
      resolution_reasoning=reasoning,
      validation=validation,
    )

  def _apply_to_state(
    self,
    state: TownState,
    macro_delta: Dict[str, float],
    policy_delta: Dict[str, float],
    price_shock: Dict[str, object],
  ) -> None:
    for field_name, delta in macro_delta.items():
      bounds = MACRO_FIELD_BOUNDS.get(field_name)
      if bounds and hasattr(state.macro, field_name):
        current = getattr(state.macro, field_name, 0.0)
        setattr(state.macro, field_name,
                max(bounds[0], min(bounds[1], current + delta)))

    for field_name, delta in policy_delta.items():
      bounds = POLICY_FIELD_BOUNDS.get(field_name)
      if bounds and hasattr(state.policy, field_name):
        current = getattr(state.policy, field_name, 0.0)
        setattr(state.policy, field_name,
                max(bounds[0], min(bounds[1], current + delta)))

    global_pct = float(price_shock.get("global_price_pct", 0.0))
    current_cpi = float(getattr(state, "consumer_price_index", 1.0) or 1.0)
    state.consumer_price_index = max(0.60, min(3.50, current_cpi * (1.0 + global_pct)))

    cat_pct = price_shock.get("category_price_pct", {})
    if not isinstance(cat_pct, dict):
      cat_pct = {}
    if not isinstance(state.category_price_multipliers, dict):
      state.category_price_multipliers = {}
    for category in SPEND_CATEGORIES:
      delta_pct = float(cat_pct.get(category, 0.0))
      base = float(state.category_price_multipliers.get(category, 1.0))
      state.category_price_multipliers[category] = max(
        0.60, min(3.50, base * (1.0 + delta_pct))
      )

  @property
  def stats(self) -> Dict[str, int]:
    return {
      "total_events": self._total_count,
      "rejected_events": self._rejection_count,
      "accepted_events": self._total_count - self._rejection_count,
    }

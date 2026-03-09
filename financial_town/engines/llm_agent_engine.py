"""LLM-backed agent decision engine (mobility + social + influence)."""

from __future__ import annotations

import json
import random
from typing import Dict, List, Optional

from financial_town.config import SimulationConfig
from financial_town.llm import AgentLLMClient
from financial_town.models import EmploymentStatus
from financial_town.simulation.state import TownState


class LLMAgentEngine:
  def __init__(self, config: SimulationConfig, rng: random.Random):
    self.config = config
    self.rng = rng
    self.client = AgentLLMClient(
      provider=config.llm_provider,
      model=config.llm_model,
      temperature=config.llm_temperature,
      max_output_tokens=config.llm_max_output_tokens,
      timeout_sec=config.llm_timeout_sec,
    )

  def _clamp(self, value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))

  def _safe_float(self, value, default: float = 0.0) -> float:
    try:
      return float(value)
    except Exception:
      return default

  def _phase(self, step: int):
    hour = step % max(4, self.config.steps_per_day)
    if 0 <= hour <= 6:
      return "home"
    if hour == 7:
      return "work"
    if 8 <= hour <= 16:
      return "work"
    if hour == 17:
      return "spending"
    if 18 <= hour <= 20:
      return "spending"
    if hour == 21:
      return "home"
    return "home"

  def _default_target_for_person(self, person, phase_target: str) -> str:
    target = phase_target
    if target == "work" and person.employment_status == EmploymentStatus.UNEMPLOYED:
      target = "spending"
    return target

  def _target_coords_and_address(self, person, place: str):
    if place == "work":
      return person.work_x, person.work_y, person.work_address
    if place == "spending":
      return person.spending_x, person.spending_y, person.spending_address
    return person.home_x, person.home_y, person.home_address

  def _move_one_step(self, curr_x: int, curr_y: int, target_x: int, target_y: int):
    x = curr_x
    y = curr_y
    if x < target_x:
      x += 1
    elif x > target_x:
      x -= 1
    elif y < target_y:
      y += 1
    elif y > target_y:
      y -= 1
    return x, y

  def _ensure_llm_stats(self, state: TownState) -> None:
    defaults = {
      "enabled": 1.0 if self.config.enable_llm_agents else 0.0,
      "calls_total": 0.0,
      "calls_step": 0.0,
      "errors_total": 0.0,
      "errors_step": 0.0,
      "prompt_tokens_total": 0.0,
      "completion_tokens_total": 0.0,
      "prompt_tokens_step": 0.0,
      "completion_tokens_step": 0.0,
      "active_agents_step": 0.0,
    }
    for key, value in defaults.items():
      if key not in state.llm_stats:
        state.llm_stats[key] = value

  def _reset_step_llm_stats(self, state: TownState) -> None:
    state.llm_stats["calls_step"] = 0.0
    state.llm_stats["errors_step"] = 0.0
    state.llm_stats["prompt_tokens_step"] = 0.0
    state.llm_stats["completion_tokens_step"] = 0.0
    state.llm_stats["active_agents_step"] = 0.0

  def _select_agent_ids(self, state: TownState) -> List[str]:
    all_ids = sorted(state.people.keys())
    if not all_ids:
      return []

    max_agents = max(1, int(self.config.llm_max_agents_per_step))
    if len(all_ids) <= max_agents:
      return all_ids

    start = (state.step * max_agents) % len(all_ids)
    selected = all_ids[start : start + max_agents]
    if len(selected) < max_agents:
      selected.extend(all_ids[: max_agents - len(selected)])
    return selected

  def _nearby_social_context(self, state: TownState, person_id: str) -> List[dict]:
    rows = []
    person = state.people[person_id]
    neighbors = state.social_graph.get(person_id, {})
    for other_id, relations in neighbors.items():
      if other_id not in state.people:
        continue
      other = state.people[other_id]
      distance = abs(person.current_x - other.current_x) + abs(person.current_y - other.current_y)
      rows.append({
        "name": other.full_name,
        "relations": relations,
        "distance": distance,
        "current_place": other.current_place,
      })
    rows.sort(key=lambda x: x["distance"])
    return rows[:6]

  def _recent_interactions(self, state: TownState, person_name: str) -> List[dict]:
    out = []
    for event in reversed(state.interaction_events):
      people = event.get("people", [])
      if person_name in people:
        out.append({
          "step": event.get("step"),
          "topic": event.get("topic"),
          "people": people,
          "relations": event.get("relations", []),
        })
      if len(out) >= 5:
        break
    return list(reversed(out))

  def _system_prompt(self) -> str:
    return (
      "You are one autonomous agent in a financial town simulation. "
      "Make a realistic next-step decision given your persona, financial status, "
      "daily routine, and social context. "
      "Return JSON only with fields: "
      "target_place, action_summary, pronunciatio, spending_multiplier, "
      "loan_stress_delta, fraud_susceptibility_delta, topic, chat, influence_on_peer. "
      "Constraints: target_place in [home,work,spending]. "
      "spending_multiplier in [0.60,1.80]. "
      "loan_stress_delta and fraud_susceptibility_delta in [-0.35,0.35]. "
      "chat can be null or {with, message}. "
      "influence_on_peer can be null or {spending_delta, loan_stress_delta, fraud_susceptibility_delta} "
      "each delta in [-0.30,0.30]."
    )

  def _user_prompt(self, context: Dict[str, object]) -> str:
    return "Context JSON:\n" + json.dumps(context, indent=2)

  def _default_movement_payload(self, person) -> Dict[str, object]:
    return {
      "movement": [int(person.current_x), int(person.current_y)],
      "pronunciatio": "🙂",
      "description": f"idle @ {person.current_place}",
      "chat": None,
    }

  def _ensure_step_movement(self, state: TownState, person_id: str) -> Dict[str, object]:
    if person_id not in state.step_movements:
      state.step_movements[person_id] = self._default_movement_payload(state.people[person_id])
    return state.step_movements[person_id]

  def _normalize_decision(self, decision: Dict[str, object], person, default_target: str) -> Dict[str, object]:
    target_place = str(decision.get("target_place", default_target)).strip().lower()
    if target_place not in {"home", "work", "spending"}:
      target_place = default_target

    action_summary = str(decision.get("action_summary", "")).strip()
    if not action_summary:
      action_summary = f"{person.current_place} routine"

    pronunciatio = str(decision.get("pronunciatio", "")).strip() or "🙂"

    spending_multiplier = self._safe_float(decision.get("spending_multiplier", 1.0), 1.0)
    spending_multiplier = self._clamp(spending_multiplier, 0.60, 1.80)

    loan_stress_delta = self._safe_float(decision.get("loan_stress_delta", 0.0), 0.0)
    loan_stress_delta = self._clamp(loan_stress_delta, -0.35, 0.35)

    fraud_sus_delta = self._safe_float(decision.get("fraud_susceptibility_delta", 0.0), 0.0)
    fraud_sus_delta = self._clamp(fraud_sus_delta, -0.35, 0.35)

    topic = str(decision.get("topic", "budget_planning")).strip().lower()
    if topic not in {"budget_planning", "spending_urge", "loan_stress", "fraud_warning", "scam_rumor"}:
      topic = "budget_planning"

    chat_payload = decision.get("chat")
    chat = None
    if isinstance(chat_payload, dict):
      chat_with = str(chat_payload.get("with", "")).strip()
      chat_message = str(chat_payload.get("message", "")).strip()
      if chat_with and chat_message:
        chat = {
          "with": chat_with,
          "message": chat_message[:140],
        }

    peer_raw = decision.get("influence_on_peer")
    influence_on_peer = None
    if isinstance(peer_raw, dict):
      influence_on_peer = {
        "spending_delta": self._clamp(self._safe_float(peer_raw.get("spending_delta", 0.0), 0.0), -0.30, 0.30),
        "loan_stress_delta": self._clamp(self._safe_float(peer_raw.get("loan_stress_delta", 0.0), 0.0), -0.30, 0.30),
        "fraud_susceptibility_delta": self._clamp(
          self._safe_float(peer_raw.get("fraud_susceptibility_delta", 0.0), 0.0),
          -0.30,
          0.30,
        ),
      }

    return {
      "target_place": target_place,
      "action_summary": action_summary[:160],
      "pronunciatio": pronunciatio[:8],
      "spending_multiplier": spending_multiplier,
      "loan_stress_delta": loan_stress_delta,
      "fraud_susceptibility_delta": fraud_sus_delta,
      "topic": topic,
      "chat": chat,
      "influence_on_peer": influence_on_peer,
    }

  def _apply_self_effects_and_movement(self, state: TownState, person_id: str, decision: Dict[str, object]) -> None:
    person = state.people[person_id]

    target_place = decision["target_place"]
    if target_place == "work" and person.employment_status == EmploymentStatus.UNEMPLOYED:
      target_place = "spending"

    target_x, target_y, target_address = self._target_coords_and_address(person, target_place)
    next_x, next_y = self._move_one_step(person.current_x, person.current_y, target_x, target_y)

    person.current_x = int(next_x)
    person.current_y = int(next_y)
    if next_x == target_x and next_y == target_y:
      person.current_place = target_place
    else:
      person.current_place = f"commuting_to_{target_place}"

    mods = state.influence_state.setdefault(person_id, {
      "spending_multiplier": 1.0,
      "loan_stress": 0.0,
      "fraud_susceptibility": 1.0,
    })
    mods["spending_multiplier"] = self._clamp(
      float(decision["spending_multiplier"]),
      0.60,
      1.80,
    )
    mods["loan_stress"] = self._clamp(
      float(mods.get("loan_stress", 0.0)) + float(decision["loan_stress_delta"]),
      0.0,
      3.0,
    )
    mods["fraud_susceptibility"] = self._clamp(
      float(mods.get("fraud_susceptibility", 1.0)) + float(decision["fraud_susceptibility_delta"]),
      0.40,
      3.0,
    )

    payload = self._ensure_step_movement(state, person_id)
    payload["movement"] = [int(next_x), int(next_y)]
    payload["pronunciatio"] = decision["pronunciatio"]
    payload["description"] = f"{decision['action_summary']} @ {target_address}"
    if "chat" not in payload:
      payload["chat"] = None

  def _add_chat(self, state: TownState, speaker_id: str, listener_id: str, message: str, topic: str, peer_inf: Optional[Dict[str, float]]) -> None:
    speaker = state.people[speaker_id]
    listener = state.people[listener_id]

    speaker_payload = self._ensure_step_movement(state, speaker_id)
    listener_payload = self._ensure_step_movement(state, listener_id)

    speaker_chat = speaker_payload.get("chat")
    if not isinstance(speaker_chat, list):
      speaker_chat = []
    listener_chat = listener_payload.get("chat")
    if not isinstance(listener_chat, list):
      listener_chat = []

    speaker_chat.append([listener.full_name, message])
    listener_chat.append([speaker.full_name, message])

    speaker_payload["chat"] = speaker_chat
    listener_payload["chat"] = listener_chat

    state.step_chat[speaker_id] = speaker_chat
    state.step_chat[listener_id] = listener_chat

    if peer_inf:
      listener_mods = state.influence_state.setdefault(listener_id, {
        "spending_multiplier": 1.0,
        "loan_stress": 0.0,
        "fraud_susceptibility": 1.0,
      })
      listener_mods["spending_multiplier"] = self._clamp(
        float(listener_mods.get("spending_multiplier", 1.0)) + float(peer_inf.get("spending_delta", 0.0)),
        0.60,
        1.80,
      )
      listener_mods["loan_stress"] = self._clamp(
        float(listener_mods.get("loan_stress", 0.0)) + float(peer_inf.get("loan_stress_delta", 0.0)),
        0.0,
        3.0,
      )
      listener_mods["fraud_susceptibility"] = self._clamp(
        float(listener_mods.get("fraud_susceptibility", 1.0)) + float(peer_inf.get("fraud_susceptibility_delta", 0.0)),
        0.40,
        3.0,
      )

    relations = state.social_graph.get(speaker_id, {}).get(listener_id, [])
    distance = abs(speaker.current_x - listener.current_x) + abs(speaker.current_y - listener.current_y)
    same_place = speaker.current_place == listener.current_place
    state.interaction_events.append({
      "step": state.step,
      "source_id": speaker_id,
      "target_id": listener_id,
      "people": [speaker.full_name, listener.full_name],
      "relations": relations,
      "topic": topic,
      "distance": distance,
      "same_place": same_place,
      "message": message,
    })

  def run_step(self, state: TownState) -> None:
    self._ensure_llm_stats(state)
    self._reset_step_llm_stats(state)

    if not state.people:
      return

    if not self.client.available:
      state.llm_stats["errors_step"] += 1.0
      state.llm_stats["errors_total"] += 1.0
      state.llm_events.append({
        "step": state.step,
        "type": "client_unavailable",
        "error": self.client.unavailable_reason,
      })
      if self.config.llm_strict_mode:
        raise RuntimeError(self.client.unavailable_reason)
      return

    selected_ids = self._select_agent_ids(state)
    state.llm_stats["active_agents_step"] = float(len(selected_ids))

    name_to_id = {person.full_name: person_id for person_id, person in state.people.items()}
    planned_chats: List[dict] = []

    phase_target = self._phase(state.step)
    system_prompt = self._system_prompt()

    for person_id in selected_ids:
      person = state.people[person_id]
      nearby = self._nearby_social_context(state, person_id)
      default_target = self._default_target_for_person(person, phase_target)

      person_mods = state.influence_state.get(person_id, {
        "spending_multiplier": 1.0,
        "loan_stress": 0.0,
        "fraud_susceptibility": 1.0,
      })

      context = {
        "step": state.step,
        "scenario": state.scenario_name,
        "phase_default": default_target,
        "person": {
          "name": person.full_name,
          "age": person.age,
          "industry": person.industry,
          "occupation": person.occupation,
          "employment_status": person.employment_status.value,
          "current_place": person.current_place,
          "checking_balance": round(person.checking_balance, 2),
          "savings_balance": round(person.savings_balance, 2),
          "credit_score": person.credit_score,
          "loan_count": len(person.loan_ids),
          "influence": {
            "spending_multiplier": round(float(person_mods.get("spending_multiplier", 1.0)), 3),
            "loan_stress": round(float(person_mods.get("loan_stress", 0.0)), 3),
            "fraud_susceptibility": round(float(person_mods.get("fraud_susceptibility", 1.0)), 3),
          },
        },
        "macro": {
          "inflation": state.macro.inflation,
          "unemployment_rate": state.macro.unemployment_rate,
          "gdp_growth": state.macro.gdp_growth,
        },
        "policy": {
          "stimulus_payment": state.policy.stimulus_payment,
          "central_bank_rate": state.policy.central_bank_rate,
          "fraud_detection_strength": state.policy.fraud_detection_strength,
        },
        "nearby_social_context": nearby,
        "recent_interactions": self._recent_interactions(state, person.full_name),
      }

      user_prompt = self._user_prompt(context)
      result = self.client.generate_json(system_prompt, user_prompt)

      state.llm_stats["calls_step"] += 1.0
      state.llm_stats["calls_total"] += 1.0
      state.llm_stats["prompt_tokens_step"] += float(result.get("prompt_tokens", 0))
      state.llm_stats["completion_tokens_step"] += float(result.get("completion_tokens", 0))
      state.llm_stats["prompt_tokens_total"] += float(result.get("prompt_tokens", 0))
      state.llm_stats["completion_tokens_total"] += float(result.get("completion_tokens", 0))

      if not result.get("ok"):
        state.llm_stats["errors_step"] += 1.0
        state.llm_stats["errors_total"] += 1.0
        state.llm_events.append({
          "step": state.step,
          "person_id": person_id,
          "person": person.full_name,
          "ok": False,
          "error": result.get("error", "unknown_error"),
        })
        if self.config.llm_strict_mode:
          raise RuntimeError(result.get("error", "LLM request failed."))
        continue

      normalized = self._normalize_decision(result.get("payload", {}), person, default_target)
      self._apply_self_effects_and_movement(state, person_id, normalized)

      state.llm_events.append({
        "step": state.step,
        "person_id": person_id,
        "person": person.full_name,
        "ok": True,
        "decision": normalized,
      })
      if self.config.llm_log_prompts:
        state.llm_events[-1]["prompt_context"] = context

      if normalized.get("chat"):
        planned_chats.append({
          "speaker_id": person_id,
          "chat": normalized.get("chat"),
          "topic": normalized.get("topic"),
          "influence_on_peer": normalized.get("influence_on_peer"),
        })

    for plan in planned_chats:
      chat = plan.get("chat")
      if not isinstance(chat, dict):
        continue
      speaker_id = plan.get("speaker_id")
      if speaker_id not in state.people:
        continue

      listener_name = str(chat.get("with", "")).strip()
      message = str(chat.get("message", "")).strip()
      if not listener_name or not message:
        continue
      if listener_name not in name_to_id:
        continue

      listener_id = name_to_id[listener_name]
      if listener_id == speaker_id:
        continue

      self._add_chat(
        state=state,
        speaker_id=speaker_id,
        listener_id=listener_id,
        message=message[:140],
        topic=str(plan.get("topic", "budget_planning")),
        peer_inf=plan.get("influence_on_peer"),
      )

    for person_id, person in state.people.items():
      self._ensure_step_movement(state, person_id)
      if person_id not in state.step_chat:
        state.step_chat[person_id] = state.step_movements[person_id].get("chat")

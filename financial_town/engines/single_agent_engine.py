"""Single free-agent engine with perception-memory-thought-action pipeline,
plus day-level planning, mood tracking, reflection, and relationship memory."""

from __future__ import annotations

import random
from typing import Dict, List, Optional

from financial_town.agent_cognition import (
  ActionModule,
  MemoryModule,
  PerceptionModule,
  PlanModule,
  ReactionModule,
  ReflectionModule,
  ThoughtModule,
)
from financial_town.config import SimulationConfig
from financial_town.models import EmploymentStatus
from financial_town.simulation.state import TownState

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


class SingleAgentEngine:
  def __init__(self, config: SimulationConfig, rng: random.Random):
    self.config = config
    self.rng = rng

    llm_kwargs = dict(
      provider=config.llm_provider,
      model=config.llm_model,
      temperature=min(1.0, max(0.0, config.llm_temperature)),
      max_output_tokens=config.llm_max_output_tokens,
      timeout_sec=config.llm_timeout_sec,
    )

    self.perception = PerceptionModule(max_nearby=14, steps_per_day=config.steps_per_day)
    self.memory = MemoryModule(max_items=2500)
    self.thought = ThoughtModule(**llm_kwargs)
    self.action = ActionModule(**llm_kwargs)
    self.planning = PlanModule(**llm_kwargs)
    self.reflection = ReflectionModule(
      provider=config.llm_provider,
      model=config.llm_model,
      temperature=min(1.0, max(0.0, config.llm_temperature + 0.1)),
      max_output_tokens=config.llm_max_output_tokens,
      timeout_sec=config.llm_timeout_sec,
    )
    self.reaction = ReactionModule(
      provider=config.llm_provider,
      model=config.llm_model,
      temperature=min(1.0, config.llm_temperature + 0.1),
      max_output_tokens=config.llm_max_output_tokens,
      timeout_sec=config.llm_timeout_sec,
    )

    self.daily_disruptions: Dict[int, Dict[str, object]] = {}
    self._last_planned_day: int = -1
    self._last_reflected_day: int = -1
    self._day_start_memory_idx: int = 0
    self._last_action: Optional[Dict[str, object]] = None
    self._action_history: List[Dict[str, object]] = []
    self._current_activity: Optional[Dict[str, object]] = None
    self._activity_steps_remaining: int = 0
    self._active_conversation: Optional[Dict[str, object]] = None
    self._today_interactions: Dict[str, int] = {}
    self._last_day_index: int = -1

  # ──────────────────────── persona summary ────────────────────────

  def _build_persona_summary(self, person, state: TownState) -> str:
    mood = state.active_agent_mood
    mood_desc = self._mood_description(mood)
    employed = person.employment_status == EmploymentStatus.EMPLOYED

    lines = [
      f"Name: {person.full_name}, Age: {person.age}",
      f"Occupation: {person.occupation} in {person.industry}"
      + (" (currently employed)" if employed else " (currently unemployed)"),
      f"Education: {person.education_level}, Spending style: {person.spending_style}",
      f"Finances: checking ${person.checking_balance:,.0f}, savings ${person.savings_balance:,.0f}, "
      f"credit score {person.credit_score}",
      f"Current mood: {mood_desc}",
    ]
    if person.loan_ids:
      lines.append(f"Has {len(person.loan_ids)} active loan(s)")

    top_relationships = list(state.active_agent_relationships.items())[:4]
    if top_relationships:
      rel_strs = []
      for name, info in top_relationships:
        sent = float(info.get("sentiment", 0.0))
        label = "close" if sent > 0.3 else ("acquaintance" if sent > -0.1 else "strained")
        rel_strs.append(f"{name} ({label})")
      lines.append(f"Key relationships: {', '.join(rel_strs)}")

    return "\n".join(lines)

  def _mood_description(self, mood: Dict[str, float]) -> str:
    v = float(mood.get("valence", 0.5))
    e = float(mood.get("energy", 0.5))
    s = float(mood.get("stress", 0.3))

    parts = []
    if v >= 0.7:
      parts.append("feeling positive")
    elif v <= 0.3:
      parts.append("feeling down")
    else:
      parts.append("emotionally neutral")

    if e >= 0.7:
      parts.append("energetic")
    elif e <= 0.3:
      parts.append("tired")

    if s >= 0.6:
      parts.append("quite stressed")
    elif s >= 0.4:
      parts.append("somewhat stressed")

    return ", ".join(parts) if parts else "calm and balanced"

  # ──────────────────────── helpers ────────────────────────

  def _clamp(self, value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))

  def _step_to_time(self, step: int):
    spd = max(4, self.config.steps_per_day)
    minutes_per_step = 1440 // spd
    step_in_day = step % spd
    total_minutes = step_in_day * minutes_per_step
    hour = total_minutes // 60
    minute = total_minutes % 60
    day_index = step // spd
    return hour, minute, day_index

  def _phase(self, step: int) -> str:
    hour, _, _ = self._step_to_time(step)
    if 0 <= hour <= 6:
      return "home"
    if 7 <= hour <= 16:
      return "work"
    if 17 <= hour <= 20:
      return "spending"
    return "home"

  def _day_index(self, step: int) -> int:
    return int(step // max(4, self.config.steps_per_day))

  def _hour_of_day(self, step: int) -> int:
    hour, _, _ = self._step_to_time(step)
    return hour

  # ──────────────────────── mood system ────────────────────────

  def _apply_mood_drift(self, state: TownState, hour: int) -> None:
    """Small natural mood changes based on time of day."""
    mood = state.active_agent_mood
    if 0 <= hour <= 5:
      mood["energy"] = self._clamp(mood["energy"] - 0.02, 0.0, 1.0)
    elif 6 <= hour <= 7:
      mood["energy"] = self._clamp(mood["energy"] + 0.08, 0.0, 1.0)
    elif 17 <= hour <= 20:
      mood["stress"] = self._clamp(mood["stress"] - 0.03, 0.0, 1.0)
    elif hour >= 21:
      mood["energy"] = self._clamp(mood["energy"] - 0.04, 0.0, 1.0)
      mood["stress"] = self._clamp(mood["stress"] - 0.02, 0.0, 1.0)

  def _apply_mood_from_thought(self, state: TownState, thought: Dict[str, object]) -> None:
    concern = float(thought.get("concern_level", 0.3))
    mood = state.active_agent_mood
    if concern >= 0.7:
      mood["stress"] = self._clamp(mood["stress"] + 0.05, 0.0, 1.0)
      mood["valence"] = self._clamp(mood["valence"] - 0.03, 0.0, 1.0)
    elif concern <= 0.2:
      mood["valence"] = self._clamp(mood["valence"] + 0.02, 0.0, 1.0)

  def _apply_mood_from_interaction(self, state: TownState, sentiment: str) -> None:
    mood = state.active_agent_mood
    if sentiment == "positive":
      mood["valence"] = self._clamp(mood["valence"] + 0.06, 0.0, 1.0)
      mood["energy"] = self._clamp(mood["energy"] + 0.03, 0.0, 1.0)
    elif sentiment == "negative":
      mood["valence"] = self._clamp(mood["valence"] - 0.05, 0.0, 1.0)
      mood["stress"] = self._clamp(mood["stress"] + 0.04, 0.0, 1.0)

  # ──────────────────────── activity block system ────────────────────────

  def _is_sleeping_phase(self, hour: int) -> bool:
    return 0 <= hour <= 5

  def _sleep_action(self) -> Dict[str, object]:
    return {
      "target_place": "home",
      "action_summary": "sleeping",
      "pronunciatio": "\U0001f634",
      "interact_with": None,
      "interact_message": None,
      "topic": "daily_life",
      "spending_multiplier": 1.0,
      "loan_stress_delta": 0.0,
      "fraud_susceptibility_delta": 0.0,
    }

  def _continue_activity_action(self) -> Dict[str, object]:
    if not self._current_activity:
      return self._sleep_action()
    return {
      "target_place": self._current_activity.get("target_place", "home"),
      "action_summary": self._current_activity.get("action_summary", "continuing activity"),
      "pronunciatio": self._current_activity.get("pronunciatio", "\U0001f642"),
      "interact_with": None,
      "interact_message": None,
      "topic": self._current_activity.get("topic", "daily_life"),
      "spending_multiplier": float(self._current_activity.get("spending_multiplier", 1.0)),
      "loan_stress_delta": 0.0,
      "fraud_susceptibility_delta": 0.0,
    }

  def _start_activity(self, action: Dict[str, object], duration_steps: int) -> None:
    self._current_activity = dict(action)
    self._activity_steps_remaining = max(1, duration_steps)

  def _tick_activity(self) -> bool:
    if self._activity_steps_remaining <= 0:
      self._current_activity = None
      return False
    self._activity_steps_remaining -= 1
    if self._activity_steps_remaining <= 0:
      self._current_activity = None
      return False
    return True

  def _estimate_duration(self, action: Dict[str, object], hour: int) -> int:
    summary = str(action.get("action_summary", "")).lower()
    duration_steps = int(action.get("duration_steps", 0))
    if duration_steps > 0:
      return min(duration_steps, 18)

    if any(w in summary for w in ["movie", "film", "show", "tv", "series"]):
      return self.rng.randint(9, 15)
    if any(w in summary for w in ["jog", "run", "exercise", "workout", "gym"]):
      return self.rng.randint(3, 5)
    if any(w in summary for w in ["cook", "dinner", "lunch", "meal"]):
      return self.rng.randint(3, 5)
    if any(w in summary for w in ["shower", "bath"]):
      return self.rng.randint(2, 3)
    if any(w in summary for w in ["read", "book"]):
      return self.rng.randint(3, 6)
    if any(w in summary for w in ["meeting", "team meeting", "standup"]):
      return self.rng.randint(3, 6)
    if any(w in summary for w in ["report", "analysis", "document", "draft"]):
      return self.rng.randint(4, 8)
    if any(w in summary for w in ["breakfast"]):
      return self.rng.randint(2, 3)
    if any(w in summary for w in ["commut", "drive", "walk to"]):
      return self.rng.randint(2, 4)
    if any(w in summary for w in ["shop", "grocery", "store", "errand"]):
      return self.rng.randint(3, 5)
    if any(w in summary for w in ["coffee", "tea", "break"]):
      return self.rng.randint(1, 2)
    if any(w in summary for w in ["sleep", "nap", "rest", "bed"]):
      return self.rng.randint(6, 12)
    return self.rng.randint(2, 4)

  def _build_action_history_context(self) -> List[Dict[str, object]]:
    recent = self._action_history[-6:]
    result = []
    for entry in recent:
      result.append({
        "time": entry.get("time", ""),
        "action": entry.get("action_summary", ""),
        "place": entry.get("target_place", ""),
        "talked_to": entry.get("interact_with"),
      })
    return result

  def _get_today_interaction_summary(self) -> Dict[str, int]:
    return dict(self._today_interactions)

  def _reset_daily_tracking(self, day_index: int) -> None:
    if day_index != self._last_day_index:
      self._today_interactions = {}
      self._last_day_index = day_index

  # ──────────────────────── relationship tracking ────────────────────────

  def _update_relationship(self, state: TownState, name: str, sentiment_delta: float, note: str = "") -> None:
    rels = state.active_agent_relationships
    if name not in rels:
      rels[name] = {"sentiment": 0.0, "interaction_count": 0, "last_note": ""}
    rel = rels[name]
    rel["sentiment"] = self._clamp(float(rel.get("sentiment", 0.0)) + sentiment_delta, -1.0, 1.0)
    rel["interaction_count"] = int(rel.get("interaction_count", 0)) + 1
    if note:
      rel["last_note"] = note[:120]

  # ──────────────────────── daily disruptions ────────────────────────

  def _build_daily_disruption(self, person, step: int) -> Dict[str, object]:
    day_index = self._day_index(step)
    if day_index in self.daily_disruptions:
      return self.daily_disruptions[day_index]

    day_of_week = day_index % 7
    is_weekend = day_of_week in {5, 6}
    employed = person.employment_status == EmploymentStatus.EMPLOYED
    roll = self.rng.random()

    event: Dict[str, object] = {
      "event_type": "none",
      "label": "normal_day",
      "reason": "no major disturbance",
      "start_hour": -1,
      "end_hour": -1,
      "target_place": "",
      "activity": "",
      "allow_interaction": True,
    }

    if not is_weekend and employed:
      if roll < 0.08:
        event = {"event_type": "sick_day", "label": "sick day", "reason": "low energy/health issue",
                 "start_hour": 8, "end_hour": 18, "target_place": "home",
                 "activity": "rest_and_recover", "allow_interaction": False}
      elif roll < 0.14:
        event = {"event_type": "doctor_visit", "label": "doctor visit", "reason": "health appointment",
                 "start_hour": 10, "end_hour": 13, "target_place": "spending",
                 "activity": "medical_appointment", "allow_interaction": False}
      elif roll < 0.22:
        event = {"event_type": "urgent_family_errand", "label": "urgent family errand",
                 "reason": "family request", "start_hour": 14, "end_hour": 18,
                 "target_place": "spending", "activity": "family_support_errand",
                 "allow_interaction": True}
      elif roll < 0.29:
        event = {"event_type": "overtime", "label": "overtime", "reason": "deadline pressure",
                 "start_hour": 17, "end_hour": 21, "target_place": "work",
                 "activity": "overtime_work", "allow_interaction": False}
      elif roll < 0.36:
        event = {"event_type": "remote_work", "label": "remote work",
                 "reason": "work from home arrangement", "start_hour": 8, "end_hour": 16,
                 "target_place": "home", "activity": "remote_work_focus",
                 "allow_interaction": False}
    elif is_weekend:
      if roll < 0.12:
        event = {"event_type": "family_visit", "label": "family visit",
                 "reason": "scheduled family time", "start_hour": 11, "end_hour": 16,
                 "target_place": "home", "activity": "family_time", "allow_interaction": True}
      elif roll < 0.22:
        event = {"event_type": "community_event", "label": "community event",
                 "reason": "local event participation", "start_hour": 10, "end_hour": 14,
                 "target_place": "spending", "activity": "community_participation",
                 "allow_interaction": True}
      elif roll < 0.33:
        event = {"event_type": "social_plan", "label": "social plan", "reason": "friends meetup",
                 "start_hour": 18, "end_hour": 22, "target_place": "spending",
                 "activity": "social_meetup", "allow_interaction": True}
    else:
      if roll < 0.15:
        event = {"event_type": "job_fair", "label": "job fair", "reason": "job search",
                 "start_hour": 9, "end_hour": 13, "target_place": "spending",
                 "activity": "job_search_event", "allow_interaction": True}
      elif roll < 0.26:
        event = {"event_type": "training_day", "label": "training day",
                 "reason": "skills improvement", "start_hour": 10, "end_hour": 15,
                 "target_place": "spending", "activity": "training_course",
                 "allow_interaction": False}

    self.daily_disruptions[day_index] = event
    return event

  def _apply_disruption_to_routine(
    self, routine: Dict[str, object], disruption: Dict[str, object], hour: int,
  ) -> Dict[str, object]:
    merged = dict(routine)
    merged["daily_event"] = disruption

    start_hour = int(disruption.get("start_hour", -1))
    end_hour = int(disruption.get("end_hour", -1))
    is_active = (
      str(disruption.get("event_type", "none")) != "none"
      and start_hour >= 0 and end_hour >= start_hour
      and start_hour <= hour <= end_hour
    )
    merged["daily_event_active"] = is_active
    merged["event_starting_now"] = bool(is_active and hour == start_hour)
    merged["event_type"] = str(disruption.get("event_type", "none"))
    merged["event_label"] = str(disruption.get("label", "normal day"))
    merged["event_reason"] = str(disruption.get("reason", ""))

    if is_active:
      event_target = str(disruption.get("target_place", "")).strip()
      if event_target in {"home", "work", "spending"}:
        merged["target_place"] = event_target
        merged["expected_place"] = event_target
      event_activity = str(disruption.get("activity", "")).strip()
      if event_activity:
        merged["activity"] = event_activity
      merged["allow_interaction"] = bool(disruption.get("allow_interaction", True))
      merged["phase"] = f"{merged.get('phase', 'day')}_with_{merged['event_type']}"

    return merged

  def _active_routine(self, person, step: int) -> Dict[str, object]:
    hour = self._hour_of_day(step)
    day_index = self._day_index(step)
    day_of_week = day_index % 7
    is_weekend = day_of_week in {5, 6}
    employed = person.employment_status == EmploymentStatus.EMPLOYED

    if 0 <= hour <= 5:
      routine = {"phase": "sleep", "target_place": "home", "activity": "sleep_or_rest",
                 "allow_interaction": False}
    elif 6 <= hour <= 7:
      routine = {"phase": "morning", "target_place": "home", "activity": "morning_prep",
                 "allow_interaction": False}
    elif (not is_weekend) and employed and 8 <= hour <= 16:
      routine = {"phase": "work_block", "target_place": "work", "activity": "work_focus",
                 "allow_interaction": hour >= 12}
    elif is_weekend and 10 <= hour <= 16:
      routine = {"phase": "weekend_day", "target_place": "spending",
                 "activity": "errands_or_leisure", "allow_interaction": True}
    elif 17 <= hour <= 20:
      routine = {"phase": "evening", "target_place": "spending",
                 "activity": "errands_social_or_personal_time", "allow_interaction": True}
    else:
      routine = {"phase": "night", "target_place": "home", "activity": "wind_down",
                 "allow_interaction": hour <= 21}

    routine["expected_place"] = routine.get("target_place", "home")
    disruption = self._build_daily_disruption(person, step)
    return self._apply_disruption_to_routine(routine, disruption, hour)

  def _align_action_with_routine(
    self, action: Dict[str, object], thought: Dict[str, object], routine: Dict[str, object],
  ) -> Dict[str, object]:
    expected_place = str(routine.get("target_place", routine.get("expected_place", "home")))
    current_place = str(action.get("target_place", "home"))
    if current_place == expected_place:
      return action

    topic = str(action.get("topic", "daily_life"))
    concern = 0.0
    try:
      concern = float(thought.get("concern_level", 0.0))
    except Exception:
      concern = 0.0

    if topic in {"loan_stress", "fraud_warning", "scam_rumor"} or concern >= 0.72:
      return action

    if bool(routine.get("daily_event_active", False)):
      strict = {"sick_day", "doctor_visit", "overtime", "remote_work", "training_day"}
      if str(routine.get("event_type", "none")) in strict:
        action["target_place"] = expected_place
        summary = str(action.get("action_summary", "follow daily event")).strip()
        action["action_summary"] = f"follow daily event: {summary}"[:180]
        if not bool(routine.get("allow_interaction", True)):
          action["interact_with"] = None
          action["interact_message"] = None
          action["topic"] = "daily_life"
        return action

    phase = str(routine.get("phase", "day"))
    if phase in {"evening", "weekend_day"} and self.rng.random() < 0.35:
      return action
    if phase not in {"evening", "weekend_day"} and self.rng.random() < 0.18:
      return action

    action["target_place"] = expected_place
    summary = str(action.get("action_summary", "follow routine")).strip()
    routine_activity = str(routine.get("activity", "routine"))
    action["action_summary"] = f"follow {routine_activity}; {summary}"[:180]
    if not bool(routine.get("allow_interaction", True)):
      action["interact_with"] = None
      action["interact_message"] = None
      action["topic"] = "daily_life"
    return action

  # ──────────────────────── movement ────────────────────────

  def _target_coords_and_address(self, person, place: str):
    if place == "work":
      return person.work_x, person.work_y, person.work_address
    if place == "spending":
      return person.spending_x, person.spending_y, person.spending_address
    return person.home_x, person.home_y, person.home_address

  def _move_one_step(self, curr_x: int, curr_y: int, target_x: int, target_y: int):
    x, y = curr_x, curr_y
    if x < target_x:
      x += 1
    elif x > target_x:
      x -= 1
    elif y < target_y:
      y += 1
    elif y > target_y:
      y -= 1
    return x, y

  def _emoji_for_place(self, place: str) -> str:
    if place == "work":
      return "\U0001f4bc"
    if place == "spending":
      return "\U0001f6cd\ufe0f"
    return "\U0001f3e0"

  def _default_target(self, person, phase_target: str) -> str:
    if phase_target == "work" and person.employment_status == EmploymentStatus.UNEMPLOYED:
      return "spending"
    return phase_target

  def _move_scripted_population(self, state: TownState, active_agent_id: str) -> None:
    phase_target = self._phase(state.step)
    for person_id, person in state.people.items():
      if person_id == active_agent_id:
        continue
      target = self._default_target(person, phase_target)
      tx, ty, address = self._target_coords_and_address(person, target)
      nx, ny = self._move_one_step(person.current_x, person.current_y, tx, ty)
      person.current_x, person.current_y = int(nx), int(ny)
      person.current_place = target if (nx == tx and ny == ty) else f"commuting_to_{target}"
      state.step_movements[person_id] = {
        "movement": [int(nx), int(ny)],
        "pronunciatio": self._emoji_for_place(target),
        "description": f"scripted routine @ {address}",
        "chat": None,
      }
      state.step_chat[person_id] = None

  def _move_active_agent(self, state: TownState, active_agent_id: str, action: Dict[str, object]) -> None:
    person = state.people[active_agent_id]
    target_place = str(action.get("target_place", "work"))
    if target_place == "work" and person.employment_status == EmploymentStatus.UNEMPLOYED:
      target_place = "spending"

    tx, ty, address = self._target_coords_and_address(person, target_place)
    nx, ny = self._move_one_step(person.current_x, person.current_y, tx, ty)
    person.current_x, person.current_y = int(nx), int(ny)
    person.current_place = target_place if (nx == tx and ny == ty) else f"commuting_to_{target_place}"

    mods = state.influence_state.setdefault(active_agent_id, {
      "spending_multiplier": 1.0, "loan_stress": 0.0, "fraud_susceptibility": 1.0,
    })
    mods["spending_multiplier"] = self._clamp(float(action.get("spending_multiplier", 1.0)), 0.60, 1.80)
    mods["loan_stress"] = self._clamp(
      float(mods.get("loan_stress", 0.0)) + float(action.get("loan_stress_delta", 0.0)), 0.0, 3.0)
    mods["fraud_susceptibility"] = self._clamp(
      float(mods.get("fraud_susceptibility", 1.0)) + float(action.get("fraud_susceptibility_delta", 0.0)), 0.40, 3.0)

    state.step_movements[active_agent_id] = {
      "movement": [int(nx), int(ny)],
      "pronunciatio": str(action.get("pronunciatio", "\U0001f642"))[:8],
      "description": f"{str(action.get('action_summary', 'agent action'))} @ {address}",
      "chat": None,
    }
    state.step_chat[active_agent_id] = None

  # ──────────────────────── LLM stats ────────────────────────

  def _ensure_llm_stats(self, state: TownState) -> None:
    for key, value in {"enabled": 1.0, "calls_total": 0.0, "calls_step": 0.0,
                       "errors_total": 0.0, "errors_step": 0.0,
                       "prompt_tokens_total": 0.0, "completion_tokens_total": 0.0,
                       "prompt_tokens_step": 0.0, "completion_tokens_step": 0.0,
                       "active_agents_step": 1.0}.items():
      if key not in state.llm_stats:
        state.llm_stats[key] = value

  def _reset_step_llm_stats(self, state: TownState) -> None:
    state.llm_stats["calls_step"] = 0.0
    state.llm_stats["errors_step"] = 0.0
    state.llm_stats["prompt_tokens_step"] = 0.0
    state.llm_stats["completion_tokens_step"] = 0.0
    state.llm_stats["active_agents_step"] = 1.0

  def _track_call(self, state: TownState, prompt_tokens: int, completion_tokens: int, ok: bool) -> None:
    state.llm_stats["calls_step"] += 1.0
    state.llm_stats["calls_total"] += 1.0
    state.llm_stats["prompt_tokens_step"] += float(prompt_tokens)
    state.llm_stats["completion_tokens_step"] += float(completion_tokens)
    state.llm_stats["prompt_tokens_total"] += float(prompt_tokens)
    state.llm_stats["completion_tokens_total"] += float(completion_tokens)
    if not ok:
      state.llm_stats["errors_step"] += 1.0
      state.llm_stats["errors_total"] += 1.0

  # ──────────────────────── interaction helpers ────────────────────────

  def _find_person_id_by_name(self, state: TownState, full_name: str) -> Optional[str]:
    target = (full_name or "").strip()
    if not target:
      return None
    for person_id, person in state.people.items():
      if person.full_name == target:
        return person_id
    return None

  def _distance(self, p1, p2) -> int:
    return abs(p1.current_x - p2.current_x) + abs(p1.current_y - p2.current_y)

  def _apply_interaction(
    self, state: TownState, active_agent_id: str, target_id: str,
    channel: str, topic: str, agent_message: str, npc_reply: str,
    reaction: Dict[str, object],
  ) -> None:
    active = state.people[active_agent_id]
    target = state.people[target_id]

    active_chat = state.step_movements[active_agent_id].get("chat")
    if not isinstance(active_chat, list):
      active_chat = []
    target_chat = state.step_movements[target_id].get("chat")
    if not isinstance(target_chat, list):
      target_chat = []

    active_chat.append([target.full_name, agent_message])
    active_chat.append([target.full_name, npc_reply])
    target_chat.append([active.full_name, agent_message])
    target_chat.append([target.full_name, npc_reply])

    state.step_movements[active_agent_id]["chat"] = active_chat
    state.step_movements[target_id]["chat"] = target_chat
    state.step_chat[active_agent_id] = active_chat
    state.step_chat[target_id] = target_chat

    target_mods = state.influence_state.setdefault(target_id, {
      "spending_multiplier": 1.0, "loan_stress": 0.0, "fraud_susceptibility": 1.0,
    })
    target_mods["spending_multiplier"] = self._clamp(
      float(target_mods.get("spending_multiplier", 1.0)) + float(reaction.get("spending_delta", 0.0)), 0.60, 1.80)
    target_mods["loan_stress"] = self._clamp(
      float(target_mods.get("loan_stress", 0.0)) + float(reaction.get("loan_stress_delta", 0.0)), 0.0, 3.0)
    target_mods["fraud_susceptibility"] = self._clamp(
      float(target_mods.get("fraud_susceptibility", 1.0)) + float(reaction.get("fraud_susceptibility_delta", 0.0)), 0.40, 3.0)

    sentiment = str(reaction.get("sentiment", "neutral"))
    self._apply_mood_from_interaction(state, sentiment)

    sent_delta = {"positive": 0.1, "neutral": 0.02, "negative": -0.08}.get(sentiment, 0.0)
    self._update_relationship(state, target.full_name, sent_delta, f"{topic}: {npc_reply[:60]}")

    state.interaction_events.append({
      "step": state.step,
      "source_id": active_agent_id,
      "target_id": target_id,
      "channel": channel,
      "people": [active.full_name, target.full_name],
      "relations": state.social_graph.get(active_agent_id, {}).get(target_id, []),
      "topic": topic,
      "distance": self._distance(active, target),
      "same_place": active.current_place == target.current_place,
      "agent_message": agent_message,
      "npc_reply": npc_reply,
      "npc_sentiment": sentiment,
    })

  # ──────────────────────── day-level: plan & reflect ────────────────────────

  def _maybe_plan_day(self, state: TownState, active_agent_id: str, perception: Dict[str, object]) -> None:
    day_index = self._day_index(state.step)
    if day_index == self._last_planned_day:
      return
    hour = self._hour_of_day(state.step)
    if hour > 1:
      return

    self._last_planned_day = day_index
    self._day_start_memory_idx = len(state.active_agent_memory)

    person = state.people[active_agent_id]
    persona_summary = self._build_persona_summary(person, state)
    is_weekend = (day_index % 7) in {5, 6}

    plan_res = self.planning.plan_day(
      persona_summary=persona_summary,
      perception=perception,
      mood=state.active_agent_mood,
      reflections=state.active_agent_reflections,
      recent_memory=self.memory.important_recent(state, limit=8),
      relationships=state.active_agent_relationships,
      day_index=day_index,
      is_weekend=is_weekend,
    )
    self._track_call(
      state,
      prompt_tokens=int(plan_res.get("prompt_tokens", 0)),
      completion_tokens=int(plan_res.get("completion_tokens", 0)),
      ok=bool(plan_res.get("ok", False)),
    )

    plan = plan_res.get("plan", self.planning.default_plan(is_weekend))
    state.active_agent_plan = plan

    day_name = DAY_NAMES[day_index % 7]
    self.memory.remember(state, {
      "step": state.step,
      "type": "plan",
      "importance": 6,
      "summary": f"Day plan ({day_name}): morning={plan['morning_goal']}, "
                 f"work={plan['work_goal']}, evening={plan['evening_goal']}, "
                 f"priority={plan['priority_concern']}",
      "topic": "daily_planning",
    })

    state.llm_events.append({
      "step": state.step, "type": "day_plan", "day_index": day_index,
      "day_name": day_name, "plan": plan,
    })

  def _maybe_reflect(self, state: TownState, active_agent_id: str) -> None:
    day_index = self._day_index(state.step)
    if day_index == self._last_reflected_day:
      return
    hour = self._hour_of_day(state.step)
    if hour != 23:
      return

    self._last_reflected_day = day_index
    person = state.people[active_agent_id]
    persona_summary = self._build_persona_summary(person, state)

    day_memories = state.active_agent_memory[self._day_start_memory_idx:]
    day_interactions = [
      e for e in state.interaction_events
      if e.get("step", -1) >= (day_index * max(4, self.config.steps_per_day))
    ]

    ref_res = self.reflection.reflect(
      persona_summary=persona_summary,
      day_memories=day_memories,
      day_plan=state.active_agent_plan,
      current_mood=state.active_agent_mood,
      interactions_today=day_interactions,
    )
    self._track_call(
      state,
      prompt_tokens=int(ref_res.get("prompt_tokens", 0)),
      completion_tokens=int(ref_res.get("completion_tokens", 0)),
      ok=bool(ref_res.get("ok", False)),
    )

    reflection = ref_res.get("reflection", self.reflection.default_reflection())

    mood_shift = reflection.get("mood_shift", {})
    mood = state.active_agent_mood
    mood["valence"] = self._clamp(mood["valence"] + float(mood_shift.get("valence_delta", 0.0)), 0.0, 1.0)
    mood["energy"] = self._clamp(mood["energy"] + float(mood_shift.get("energy_delta", 0.0)), 0.0, 1.0)
    mood["stress"] = self._clamp(mood["stress"] + float(mood_shift.get("stress_delta", 0.0)), 0.0, 1.0)

    for ru in reflection.get("relationship_updates", []):
      self._update_relationship(state, ru["name"], ru["sentiment_delta"], ru.get("note", ""))

    ref_record = {
      "step": state.step,
      "day_index": day_index,
      "reflection": reflection["reflection"],
      "lessons": reflection.get("lessons", []),
      "mood_after": dict(mood),
    }
    state.active_agent_reflections.append(ref_record)

    self.memory.remember(state, {
      "step": state.step,
      "type": "reflection",
      "importance": 9,
      "summary": reflection["reflection"],
      "topic": "daily_reflection",
    })

    state.llm_events.append({
      "step": state.step, "type": "daily_reflection",
      "day_index": day_index, "reflection": ref_record,
    })

  # ──────────────────────── multi-turn conversation ────────────────────────

  def _run_conversation(
    self, state: TownState, active_agent_id: str,
    target_id: str, opening_message: str, topic: str,
    max_turns: int = 6,
  ) -> List[Dict[str, str]]:
    active = state.people[active_agent_id]
    target = state.people[target_id]
    dist = self._distance(active, target)
    channel = "in_person" if (dist <= 3 or active.current_place == target.current_place) else "phone"

    listener_profile = {
      "name": target.full_name,
      "industry": target.industry,
      "occupation": target.occupation,
      "employment_status": target.employment_status.value,
      "spending_style": target.spending_style,
      "credit_score": target.credit_score,
      "checking_balance": round(target.checking_balance, 2),
      "savings_balance": round(target.savings_balance, 2),
      "interaction_channel": channel,
    }

    history: List[Dict[str, str]] = []
    all_messages: List[List[str]] = []
    current_message = opening_message
    last_sentiment = "neutral"

    for turn in range(max_turns):
      reaction_res = self.reaction.react(
        speaker_name=active.full_name,
        listener_profile=listener_profile,
        message=current_message,
        topic=topic,
        conversation_history=history if history else None,
      )
      self._track_call(
        state,
        prompt_tokens=int(reaction_res.get("prompt_tokens", 0)),
        completion_tokens=int(reaction_res.get("completion_tokens", 0)),
        ok=bool(reaction_res.get("ok", False)),
      )
      if not reaction_res.get("ok"):
        if self.config.llm_strict_mode and turn == 0:
          raise RuntimeError(reaction_res.get("error", "reaction_generation_failed"))
        break

      reaction = reaction_res.get("reaction", self.reaction.default_reaction())
      reply = str(reaction.get("reply", "I see.")).strip()
      last_sentiment = str(reaction.get("sentiment", "neutral"))

      history.append({"speaker": active.full_name, "text": current_message})
      history.append({"speaker": target.full_name, "text": reply})
      all_messages.append([target.full_name, current_message])
      all_messages.append([target.full_name, reply])

      wants_continue = bool(reaction.get("wants_to_continue", False))
      if not wants_continue or turn >= max_turns - 1:
        break

      follow_up = self._generate_follow_up(
        state, active_agent_id, target.full_name, history, topic,
      )
      if not follow_up:
        break
      current_message = follow_up

    if all_messages:
      self._apply_interaction(
        state=state, active_agent_id=active_agent_id, target_id=target_id,
        channel=channel, topic=topic,
        agent_message=opening_message,
        npc_reply=history[-1]["text"] if history else "I see.",
        reaction={"sentiment": last_sentiment, "spending_delta": 0.0,
                   "loan_stress_delta": 0.0, "fraud_susceptibility_delta": 0.0},
      )
      state.step_movements[active_agent_id]["chat"] = all_messages
      if target_id in state.step_movements:
        state.step_movements[target_id]["chat"] = all_messages

      convo_summary = "; ".join(
        f"{h['speaker']}: {h['text'][:60]}" for h in history[:6]
      )
      self.memory.remember(state, {
        "step": state.step, "type": "interaction", "importance": 7,
        "summary": f"Conversation with {target.full_name} ({channel}, {len(history)//2} turns): {convo_summary}"[:400],
        "topic": topic, "counterparty": target.full_name,
      })
      name = target.full_name
      self._today_interactions[name] = self._today_interactions.get(name, 0) + 1

    return history

  def _generate_follow_up(
    self, state: TownState, active_agent_id: str,
    target_name: str, history: List[Dict[str, str]], topic: str,
  ) -> Optional[str]:
    person = state.people[active_agent_id]
    persona_summary = self._build_persona_summary(person, state)

    prompt = (
      f"You are {person.full_name}. You are in a conversation with {target_name} "
      f"about {topic}. Here is the conversation so far:\n"
    )
    for h in history[-6:]:
      prompt += f"  {h['speaker']}: {h['text']}\n"
    prompt += (
      "\nWhat do you say next? If the conversation has reached a natural end, "
      "return JSON: {\"continue\": false}. "
      "If you want to keep talking, return JSON: {\"continue\": true, \"message\": \"your reply here\"}."
    )

    result = self.thought.client.generate_json(
      f"You are {person.full_name}. {persona_summary}", prompt,
    )
    self._track_call(
      state,
      prompt_tokens=int(result.get("prompt_tokens", 0)),
      completion_tokens=int(result.get("completion_tokens", 0)),
      ok=bool(result.get("ok", False)),
    )
    if not result.get("ok"):
      return None
    payload = result.get("payload", {})
    if not payload.get("continue", False):
      return None
    msg = str(payload.get("message", "")).strip()
    return msg[:180] if msg else None

  # ──────────────────────── main step ────────────────────────

  def run_step(self, state: TownState) -> None:
    active_agent_id = state.active_agent_id
    if not active_agent_id or active_agent_id not in state.people:
      raise RuntimeError("SingleAgentEngine requires valid state.active_agent_id.")

    if not self.thought.client.available:
      raise RuntimeError(self.thought.client.unavailable_reason)
    if not self.action.client.available:
      raise RuntimeError(self.action.client.unavailable_reason)
    if not self.reaction.client.available:
      raise RuntimeError(self.reaction.client.unavailable_reason)

    self._ensure_llm_stats(state)
    self._reset_step_llm_stats(state)

    person = state.people[active_agent_id]
    hour = self._hour_of_day(state.step)
    day_index = self._day_index(state.step)
    self._reset_daily_tracking(day_index)

    self._move_scripted_population(state, active_agent_id)
    routine = self._active_routine(person, state.step)
    state.active_agent_runtime_context = {"routine_context": routine}

    self._apply_mood_drift(state, hour)

    if bool(routine.get("event_starting_now", False)):
      event_label = str(routine.get("event_label", "daily event"))
      event_reason = str(routine.get("event_reason", ""))
      event_type = str(routine.get("event_type", "daily_event"))
      event_activity = str(routine.get("activity", "routine adjustment"))
      summary = f"Daily event started: {event_label} ({event_activity})"
      if event_reason:
        summary = f"{summary}; reason: {event_reason}"
      self.memory.remember(state, {
        "step": state.step, "type": "life_event", "importance": 6,
        "summary": summary[:240], "topic": event_type,
      })

    # SLEEP PHASE: no LLM calls, just sleep
    if self._is_sleeping_phase(hour):
      action = self._sleep_action()
      self._move_active_agent(state, active_agent_id, action)
      self._last_action = action
      self._current_activity = None
      self._activity_steps_remaining = 0
      state.llm_events.append({
        "step": state.step, "type": "single_agent_cycle",
        "active_agent": person.full_name, "routine": routine,
        "thought": {"thought": "sleeping", "objective": "rest", "concern_level": 0.0, "focus_keywords": ["sleep"]},
        "action": action, "mood": dict(state.active_agent_mood),
      })
      self._maybe_reflect(state, active_agent_id)
      return

    # CONTINUING ACTIVITY: if agent is in the middle of a multi-step activity
    if self._tick_activity():
      action = self._continue_activity_action()
      self._move_active_agent(state, active_agent_id, action)
      self._last_action = action
      state.llm_events.append({
        "step": state.step, "type": "single_agent_cycle",
        "active_agent": person.full_name, "routine": routine,
        "thought": {"thought": f"continuing: {action.get('action_summary', '')}",
                     "objective": "continue_activity", "concern_level": 0.1, "focus_keywords": ["focus"]},
        "action": action, "mood": dict(state.active_agent_mood),
      })
      self._maybe_reflect(state, active_agent_id)
      return

    # NEW DECISION NEEDED -- run full perception-thought-action pipeline
    perception = self.perception.perceive(state, active_agent_id)
    persona_summary = self._build_persona_summary(person, state)
    self._maybe_plan_day(state, active_agent_id, perception)

    recent_memory = self.memory.recent(state, limit=10)
    important_memory = self.memory.important_recent(state, limit=4, min_importance=5)
    combined_memory = important_memory + [m for m in recent_memory if m not in important_memory]

    last_action_context = None
    if self._last_action:
      last_action_context = {
        "action": self._last_action.get("action_summary", ""),
        "place": self._last_action.get("target_place", ""),
        "talked_to": self._last_action.get("interact_with"),
      }

    _, minute, _ = self._step_to_time(state.step)
    time_str = f"{hour:02d}:{minute:02d}"

    thought_res = self.thought.think(
      perception, combined_memory,
      mood=state.active_agent_mood,
      plan=state.active_agent_plan,
      persona_summary=persona_summary,
      recent_reflections=state.active_agent_reflections,
      last_action=last_action_context,
    )
    self._track_call(
      state,
      prompt_tokens=int(thought_res.get("prompt_tokens", 0)),
      completion_tokens=int(thought_res.get("completion_tokens", 0)),
      ok=bool(thought_res.get("ok", False)),
    )
    if not thought_res.get("ok") and self.config.llm_strict_mode:
      raise RuntimeError(thought_res.get("error", "thought_generation_failed"))

    thought = thought_res.get("thought", self.thought.default_thought(perception))
    self._apply_mood_from_thought(state, thought)
    state.active_agent_thoughts.append({"step": state.step, **thought})
    self.memory.remember(state, {
      "step": state.step, "type": "thought",
      "summary": thought.get("thought", ""), "topic": thought.get("objective", ""),
    })

    action_history = self._build_action_history_context()
    today_interactions = self._get_today_interaction_summary()

    action_res = self.action.decide(
      perception, thought,
      mood=state.active_agent_mood,
      plan=state.active_agent_plan,
      persona_summary=persona_summary,
      relationships=state.active_agent_relationships,
      last_action=last_action_context,
      action_history=action_history,
      today_interactions=today_interactions,
    )
    self._track_call(
      state,
      prompt_tokens=int(action_res.get("prompt_tokens", 0)),
      completion_tokens=int(action_res.get("completion_tokens", 0)),
      ok=bool(action_res.get("ok", False)),
    )
    if not action_res.get("ok") and self.config.llm_strict_mode:
      raise RuntimeError(action_res.get("error", "action_generation_failed"))

    action = action_res.get("action", self.action.default_action(thought, hour))
    action = self._align_action_with_routine(action, thought, routine)
    self._move_active_agent(state, active_agent_id, action)
    self._last_action = action

    self._action_history.append({
      "time": time_str,
      "step": state.step,
      **action,
    })
    if len(self._action_history) > 20:
      self._action_history = self._action_history[-20:]

    duration = self._estimate_duration(action, hour)
    if duration > 1:
      self._start_activity(action, duration)

    self.memory.remember(state, {
      "step": state.step, "type": "action", "importance": 4,
      "summary": f"[{time_str}] {action.get('action_summary', '')}",
      "topic": action.get("topic", "daily_life"),
    })

    # Conversation (multi-turn)
    target_name = action.get("interact_with")
    message = action.get("interact_message")
    topic = str(action.get("topic", "daily_life"))

    if target_name and message and bool(routine.get("allow_interaction", True)):
      target_id = self._find_person_id_by_name(state, str(target_name))
      if target_id and target_id != active_agent_id:
        self._run_conversation(
          state, active_agent_id, target_id,
          str(message), topic, max_turns=6,
        )

    self._maybe_reflect(state, active_agent_id)

    state.llm_events.append({
      "step": state.step,
      "type": "single_agent_cycle",
      "active_agent": person.full_name,
      "routine": routine,
      "thought": thought,
      "action": action,
      "mood": dict(state.active_agent_mood),
    })

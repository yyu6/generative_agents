"""Perception module for active agent context assembly."""

from __future__ import annotations

from typing import Dict, List

HOUR_LABELS = {
  0: "midnight", 1: "1 AM", 2: "2 AM", 3: "3 AM", 4: "4 AM", 5: "5 AM",
  6: "6 AM", 7: "7 AM", 8: "8 AM", 9: "9 AM", 10: "10 AM", 11: "11 AM",
  12: "noon", 13: "1 PM", 14: "2 PM", 15: "3 PM", 16: "4 PM", 17: "5 PM",
  18: "6 PM", 19: "7 PM", 20: "8 PM", 21: "9 PM", 22: "10 PM", 23: "11 PM",
}
DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


class PerceptionModule:
  def __init__(self, max_nearby: int = 12, steps_per_day: int = 24):
    self.max_nearby = max(3, int(max_nearby))
    self.steps_per_day = max(4, int(steps_per_day))

  def _distance(self, p1, p2) -> int:
    return abs(p1.current_x - p2.current_x) + abs(p1.current_y - p2.current_y)

  def _step_to_time(self, step: int):
    minutes_per_step = 1440 // self.steps_per_day
    step_in_day = step % self.steps_per_day
    total_minutes = step_in_day * minutes_per_step
    hour = total_minutes // 60
    minute = total_minutes % 60
    day_index = step // self.steps_per_day
    return hour, minute, day_index

  def _routine_expectation(self, active, step: int) -> Dict[str, object]:
    hour, minute, day_index = self._step_to_time(step)
    day_of_week = day_index % 7
    is_weekend = day_of_week in {5, 6}
    employed = str(active.employment_status.value) == "employed"

    if 0 <= hour <= 5:
      return {"phase": "sleep", "expected_place": "home", "activity": "sleep_or_rest",
              "allow_interaction": False, "is_weekend": is_weekend}
    if 6 <= hour <= 7:
      return {"phase": "morning", "expected_place": "home", "activity": "morning_prep",
              "allow_interaction": False, "is_weekend": is_weekend}
    if (not is_weekend) and employed and 8 <= hour <= 16:
      return {"phase": "work_block", "expected_place": "work", "activity": "work_focus",
              "allow_interaction": hour >= 12, "is_weekend": is_weekend}
    if is_weekend and 10 <= hour <= 16:
      return {"phase": "weekend_day", "expected_place": "spending",
              "activity": "errands_or_leisure", "allow_interaction": True, "is_weekend": is_weekend}
    if 17 <= hour <= 20:
      return {"phase": "evening", "expected_place": "spending",
              "activity": "errands_social_or_personal_time", "allow_interaction": True,
              "is_weekend": is_weekend}
    return {"phase": "night", "expected_place": "home", "activity": "wind_down",
            "allow_interaction": hour <= 21, "is_weekend": is_weekend}

  def perceive(self, state, active_agent_id: str) -> Dict[str, object]:
    if active_agent_id not in state.people:
      return {"error": f"active_agent_id not found: {active_agent_id}"}

    active = state.people[active_agent_id]
    runtime_ctx = getattr(state, "active_agent_runtime_context", {}) or {}
    routine = runtime_ctx.get("routine_context", None) if isinstance(runtime_ctx, dict) else None
    if not isinstance(routine, dict):
      routine = self._routine_expectation(active, state.step)

    nearby: List[dict] = []
    for person_id, person in state.people.items():
      if person_id == active_agent_id:
        continue
      nearby.append({
        "person_id": person_id,
        "name": person.full_name,
        "distance": self._distance(active, person),
        "current_place": person.current_place,
        "industry": person.industry,
        "occupation": person.occupation,
        "employment_status": person.employment_status.value,
        "checking_balance": round(person.checking_balance, 2),
        "savings_balance": round(person.savings_balance, 2),
      })

    nearby.sort(key=lambda x: x["distance"])
    nearby = nearby[: self.max_nearby]

    social_neighbors = []
    town_directory = []
    for peer_id, relations in state.social_graph.get(active_agent_id, {}).items():
      if peer_id not in state.people:
        continue
      peer = state.people[peer_id]
      social_neighbors.append({
        "person_id": peer_id,
        "name": peer.full_name,
        "relations": relations,
        "distance": self._distance(active, peer),
        "current_place": peer.current_place,
      })
      town_directory.append({
        "person_id": peer_id,
        "name": peer.full_name,
        "industry": peer.industry,
        "occupation": peer.occupation,
        "employment_status": peer.employment_status.value,
      })
    social_neighbors.sort(key=lambda x: x["distance"])
    town_directory.sort(key=lambda x: x["name"])

    hour, minute, day_index = self._step_to_time(state.step)
    day_of_week = day_index % 7

    return {
      "step": state.step,
      "time": {
        "hour_of_day": hour,
        "minute_of_hour": minute,
        "hour_label": HOUR_LABELS.get(hour, str(hour)),
        "time_display": f"{hour:02d}:{minute:02d}",
        "day_index": day_index,
        "day_of_week_index": day_of_week,
        "day_name": DAY_NAMES[day_of_week],
        "is_weekend": day_of_week in {5, 6},
      },
      "routine_context": routine,
      "self": {
        "person_id": active_agent_id,
        "name": active.full_name,
        "age": active.age,
        "industry": active.industry,
        "occupation": active.occupation,
        "employment_status": active.employment_status.value,
        "spending_style": active.spending_style,
        "checking_balance": round(active.checking_balance, 2),
        "savings_balance": round(active.savings_balance, 2),
        "credit_score": active.credit_score,
        "current_place": active.current_place,
        "coordinates": [active.current_x, active.current_y],
      },
      "macro": {
        "gdp_growth": round(state.macro.gdp_growth, 5),
        "inflation": round(state.macro.inflation, 5),
        "unemployment_rate": round(state.macro.unemployment_rate, 5),
        "housing_growth": round(state.macro.housing_growth, 5),
      },
      "policy": {
        "layoff_rate": state.policy.layoff_rate,
        "stimulus_payment": state.policy.stimulus_payment,
        "central_bank_rate": state.policy.central_bank_rate,
        "fraud_detection_strength": state.policy.fraud_detection_strength,
      },
      "prices": {
        "consumer_price_index": round(float(getattr(state, "consumer_price_index", 1.0) or 1.0), 5),
        "category_price_multipliers": {
          str(k): round(float(v), 5)
          for k, v in (getattr(state, "category_price_multipliers", {}) or {}).items()
        },
      },
      "nearby_people": nearby,
      "social_neighbors": social_neighbors[: self.max_nearby],
      "town_directory": town_directory,
      "recent_macro_events": state.macro_events[-6:],
    }

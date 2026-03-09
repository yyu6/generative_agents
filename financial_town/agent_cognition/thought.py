"""Thought module: derive reflective thought from perception + memory + mood + plan."""

from __future__ import annotations

import json
from typing import Dict, List

from financial_town.llm import AgentLLMClient

HOUR_LABELS = {
  0: "midnight", 1: "1 AM", 2: "2 AM", 3: "3 AM", 4: "4 AM", 5: "5 AM",
  6: "6 AM", 7: "7 AM", 8: "8 AM", 9: "9 AM", 10: "10 AM", 11: "11 AM",
  12: "noon", 13: "1 PM", 14: "2 PM", 15: "3 PM", 16: "4 PM", 17: "5 PM",
  18: "6 PM", 19: "7 PM", 20: "8 PM", 21: "9 PM", 22: "10 PM", 23: "11 PM",
}

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


class ThoughtModule:
  def __init__(
    self,
    provider: str,
    model: str,
    temperature: float,
    max_output_tokens: int,
    timeout_sec: int,
  ):
    self.client = AgentLLMClient(
      provider=provider,
      model=model,
      temperature=temperature,
      max_output_tokens=max_output_tokens,
      timeout_sec=timeout_sec,
    )

  def system_prompt(self, persona_summary: str) -> str:
    return (
      f"You are the inner voice of the following person:\n{persona_summary}\n\n"
      "Produce a brief, natural internal thought (1-2 sentences) reflecting what "
      "you'd actually think right now given what you just did and the time of day. "
      "Be specific and grounded: reference your actual situation, not abstract plans. "
      "If you notice macro-economic events (rate changes, inflation), think about "
      "how they affect your finances or daily life. "
      "Return JSON with keys: thought, objective, concern_level, focus_keywords. "
      "Constraints: concern_level in [0,1], focus_keywords is array of 2-6 short strings."
    )

  def build_user_prompt(
    self,
    perception: Dict[str, object],
    memory_items: List[Dict[str, object]],
    mood: Dict[str, float],
    plan: Dict[str, object],
    recent_reflections: List[Dict[str, object]],
    last_action: Dict[str, object] | None = None,
  ) -> str:
    time_info = perception.get("time", {})
    hour = int(time_info.get("hour_of_day", 0))
    day_idx = int(time_info.get("day_of_week_index", 0))
    minute = int(time_info.get("minute_of_hour", 0))

    payload = {
      "current_time": f"{HOUR_LABELS.get(hour, str(hour))}:{minute:02d}, {DAY_NAMES[day_idx % 7]}",
      "what_you_just_did": last_action or "nothing yet (start of day)",
      "perception": perception,
      "mood": mood,
      "todays_plan": plan,
      "recent_reflections": recent_reflections[-2:],
      "memory": memory_items,
    }
    return "What are you thinking right now? (must differ from your last thought)\n" + json.dumps(payload, indent=2)

  def default_thought(self, perception: Dict[str, object]) -> Dict[str, object]:
    time_info = perception.get("time", {})
    hour = int(time_info.get("hour_of_day", 0))

    if 0 <= hour <= 5:
      thought_text = "I should get some sleep. Tomorrow is another day."
      objective = "rest_and_recover"
    elif 6 <= hour <= 7:
      thought_text = "Time to get up and get ready for the day ahead."
      objective = "morning_preparation"
    elif 8 <= hour <= 16:
      thought_text = "I need to stay focused and productive today."
      objective = "work_and_productivity"
    elif 17 <= hour <= 20:
      thought_text = "Work is done. I should take care of errands or spend some time with people."
      objective = "evening_errands_or_social"
    else:
      thought_text = "Getting late. Time to wind down and head home."
      objective = "wind_down"

    return {
      "thought": thought_text,
      "objective": objective,
      "concern_level": 0.3,
      "focus_keywords": ["daily_life", "wellbeing", "budget"],
    }

  def think(
    self,
    perception: Dict[str, object],
    memory_items: List[Dict[str, object]],
    mood: Dict[str, float] | None = None,
    plan: Dict[str, object] | None = None,
    persona_summary: str = "",
    recent_reflections: List[Dict[str, object]] | None = None,
    last_action: Dict[str, object] | None = None,
  ) -> Dict[str, object]:
    if mood is None:
      mood = {"valence": 0.6, "energy": 0.7, "stress": 0.2}
    if plan is None:
      plan = {}
    if recent_reflections is None:
      recent_reflections = []

    result = self.client.generate_json(
      self.system_prompt(persona_summary),
      self.build_user_prompt(perception, memory_items, mood, plan, recent_reflections, last_action),
    )
    if not result.get("ok"):
      return {
        "ok": False,
        "error": result.get("error", "thought_generation_failed"),
        "thought": self.default_thought(perception),
        "prompt_tokens": 0,
        "completion_tokens": 0,
      }

    payload = result.get("payload", {})
    concern_level = payload.get("concern_level", 0.35)
    try:
      concern_level = float(concern_level)
    except Exception:
      concern_level = 0.35
    concern_level = max(0.0, min(1.0, concern_level))

    focus_keywords = payload.get("focus_keywords", [])
    if not isinstance(focus_keywords, list):
      focus_keywords = []
    focus_keywords = [str(x)[:32] for x in focus_keywords if str(x).strip()][:6]

    thought = {
      "thought": str(payload.get("thought", "I should continue my plan.")).strip()[:300],
      "objective": str(payload.get("objective", "continue_plan")).strip()[:80],
      "concern_level": concern_level,
      "focus_keywords": focus_keywords or ["budget", "social"],
    }

    return {
      "ok": True,
      "thought": thought,
      "prompt_tokens": int(result.get("prompt_tokens", 0)),
      "completion_tokens": int(result.get("completion_tokens", 0)),
    }

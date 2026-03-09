"""Planning module: generate a day-level plan at the start of each day."""

from __future__ import annotations

import json
from typing import Dict, List

from financial_town.llm import AgentLLMClient

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


class PlanModule:
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
      f"You are the planning module of the following person:\n{persona_summary}\n\n"
      "Given your current situation (finances, mood, relationships, recent events, "
      "and today's schedule), create a realistic day plan. "
      "Think about what a real person in this situation would want to accomplish today. "
      "Return JSON only with keys: morning_goal, work_goal, evening_goal, "
      "social_intention (null or a short description of who you want to talk to and why), "
      "spending_intention (frugal/normal/treat_myself), priority_concern (one sentence)."
    )

  def user_prompt(
    self,
    perception: Dict[str, object],
    mood: Dict[str, float],
    reflections: List[Dict[str, object]],
    recent_memory: List[Dict[str, object]],
    relationships: Dict[str, Dict[str, object]],
    day_name: str,
    is_weekend: bool,
  ) -> str:
    payload = {
      "day": day_name,
      "is_weekend": is_weekend,
      "self": perception.get("self", {}),
      "mood": mood,
      "recent_reflections": reflections[-3:],
      "key_memories": recent_memory[-8:],
      "close_relationships": dict(list(relationships.items())[:6]),
      "macro_economy": perception.get("macro", {}),
      "prices": perception.get("prices", {}),
      "routine_context": perception.get("routine_context", {}),
    }
    return "Plan your day:\n" + json.dumps(payload, indent=2)

  def default_plan(self, is_weekend: bool) -> Dict[str, object]:
    if is_weekend:
      return {
        "morning_goal": "sleep in and have a relaxed breakfast",
        "work_goal": "no work today, catch up on personal errands",
        "evening_goal": "relax at home or meet friends",
        "social_intention": None,
        "spending_intention": "normal",
        "priority_concern": "enjoy the weekend while being mindful of budget",
      }
    return {
      "morning_goal": "get ready and commute to work on time",
      "work_goal": "focus on tasks and be productive",
      "evening_goal": "unwind, maybe run errands or socialize",
      "social_intention": None,
      "spending_intention": "normal",
      "priority_concern": "stay on top of work while managing finances",
    }

  def plan_day(
    self,
    persona_summary: str,
    perception: Dict[str, object],
    mood: Dict[str, float],
    reflections: List[Dict[str, object]],
    recent_memory: List[Dict[str, object]],
    relationships: Dict[str, Dict[str, object]],
    day_index: int,
    is_weekend: bool,
  ) -> Dict[str, object]:
    day_name = DAY_NAMES[day_index % 7]
    result = self.client.generate_json(
      self.system_prompt(persona_summary),
      self.user_prompt(perception, mood, reflections, recent_memory, relationships, day_name, is_weekend),
    )
    if not result.get("ok"):
      return {
        "ok": False,
        "error": result.get("error", "plan_generation_failed"),
        "plan": self.default_plan(is_weekend),
        "prompt_tokens": 0,
        "completion_tokens": 0,
      }

    payload = result.get("payload", {})
    plan = {
      "morning_goal": str(payload.get("morning_goal", "prepare for the day")).strip()[:200],
      "work_goal": str(payload.get("work_goal", "be productive")).strip()[:200],
      "evening_goal": str(payload.get("evening_goal", "relax")).strip()[:200],
      "social_intention": None,
      "spending_intention": str(payload.get("spending_intention", "normal")).strip().lower(),
      "priority_concern": str(payload.get("priority_concern", "manage daily life")).strip()[:200],
    }
    si = payload.get("social_intention")
    if si is not None:
      plan["social_intention"] = str(si).strip()[:200] or None
    if plan["spending_intention"] not in {"frugal", "normal", "treat_myself"}:
      plan["spending_intention"] = "normal"

    return {
      "ok": True,
      "plan": plan,
      "prompt_tokens": int(result.get("prompt_tokens", 0)),
      "completion_tokens": int(result.get("completion_tokens", 0)),
    }

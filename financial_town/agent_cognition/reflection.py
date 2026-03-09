"""Reflection module: end-of-day synthesis of experiences into higher-level insights."""

from __future__ import annotations

import json
from typing import Dict, List

from financial_town.llm import AgentLLMClient


class ReflectionModule:
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
      f"You are the reflective consciousness of the following person:\n{persona_summary}\n\n"
      "At the end of this day, review what happened and produce a personal reflection. "
      "Synthesize specific events into higher-level insights about your life, finances, "
      "relationships, and emotional state. This is your private internal monologue. "
      "Return JSON only with keys: reflection (2-4 sentences of genuine self-reflection), "
      "lessons (array of 1-3 short takeaway strings), "
      "mood_shift (object with valence_delta, energy_delta, stress_delta each in [-0.3, 0.3]), "
      "relationship_updates (array of objects with name, sentiment_delta in [-0.3, 0.3], note)."
    )

  def user_prompt(
    self,
    day_memories: List[Dict[str, object]],
    day_plan: Dict[str, object],
    current_mood: Dict[str, float],
    interactions_today: List[Dict[str, object]],
  ) -> str:
    payload = {
      "todays_plan": day_plan,
      "current_mood": current_mood,
      "todays_events": day_memories[-20:],
      "todays_interactions": interactions_today[-8:],
    }
    return "Reflect on today:\n" + json.dumps(payload, indent=2)

  def default_reflection(self) -> Dict[str, object]:
    return {
      "reflection": "Today was an ordinary day. Nothing particularly eventful happened.",
      "lessons": ["keep going with my routine"],
      "mood_shift": {"valence_delta": 0.0, "energy_delta": -0.05, "stress_delta": 0.0},
      "relationship_updates": [],
    }

  def reflect(
    self,
    persona_summary: str,
    day_memories: List[Dict[str, object]],
    day_plan: Dict[str, object],
    current_mood: Dict[str, float],
    interactions_today: List[Dict[str, object]],
  ) -> Dict[str, object]:
    result = self.client.generate_json(
      self.system_prompt(persona_summary),
      self.user_prompt(day_memories, day_plan, current_mood, interactions_today),
    )
    if not result.get("ok"):
      return {
        "ok": False,
        "error": result.get("error", "reflection_generation_failed"),
        "reflection": self.default_reflection(),
        "prompt_tokens": 0,
        "completion_tokens": 0,
      }

    payload = result.get("payload", {})

    def _clamp(v, lo, hi):
      try:
        return max(lo, min(hi, float(v)))
      except (ValueError, TypeError):
        return 0.0

    mood_shift_raw = payload.get("mood_shift", {})
    if not isinstance(mood_shift_raw, dict):
      mood_shift_raw = {}
    mood_shift = {
      "valence_delta": _clamp(mood_shift_raw.get("valence_delta", 0.0), -0.3, 0.3),
      "energy_delta": _clamp(mood_shift_raw.get("energy_delta", 0.0), -0.3, 0.3),
      "stress_delta": _clamp(mood_shift_raw.get("stress_delta", 0.0), -0.3, 0.3),
    }

    lessons_raw = payload.get("lessons", [])
    if not isinstance(lessons_raw, list):
      lessons_raw = []
    lessons = [str(x).strip()[:120] for x in lessons_raw if str(x).strip()][:3]

    rel_updates_raw = payload.get("relationship_updates", [])
    if not isinstance(rel_updates_raw, list):
      rel_updates_raw = []
    relationship_updates = []
    for ru in rel_updates_raw[:5]:
      if not isinstance(ru, dict):
        continue
      name = str(ru.get("name", "")).strip()
      if not name:
        continue
      relationship_updates.append({
        "name": name,
        "sentiment_delta": _clamp(ru.get("sentiment_delta", 0.0), -0.3, 0.3),
        "note": str(ru.get("note", "")).strip()[:120],
      })

    reflection = {
      "reflection": str(payload.get("reflection", "Today was an ordinary day.")).strip()[:500],
      "lessons": lessons or ["keep going"],
      "mood_shift": mood_shift,
      "relationship_updates": relationship_updates,
    }

    return {
      "ok": True,
      "reflection": reflection,
      "prompt_tokens": int(result.get("prompt_tokens", 0)),
      "completion_tokens": int(result.get("completion_tokens", 0)),
    }

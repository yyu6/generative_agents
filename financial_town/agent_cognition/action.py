"""Action module: decide next action and optional social interaction."""

from __future__ import annotations

import json
from typing import Dict, List

from financial_town.llm import AgentLLMClient

ALLOWED_TOPICS = {
  "daily_life",
  "family",
  "friendship",
  "hobby",
  "health",
  "community",
  "local_news",
  "budget_planning",
  "shopping",
  "loan_stress",
  "fraud_warning",
  "scam_rumor",
}

HOUR_LABELS = {
  0: "midnight", 1: "1 AM", 2: "2 AM", 3: "3 AM", 4: "4 AM", 5: "5 AM",
  6: "6 AM", 7: "7 AM", 8: "8 AM", 9: "9 AM", 10: "10 AM", 11: "11 AM",
  12: "noon", 13: "1 PM", 14: "2 PM", 15: "3 PM", 16: "4 PM", 17: "5 PM",
  18: "6 PM", 19: "7 PM", 20: "8 PM", 21: "9 PM", 22: "10 PM", 23: "11 PM",
}

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


class ActionModule:
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
      f"You are the action planner of the following person:\n{persona_summary}\n\n"
      "Choose the next activity. Each activity has a DURATION (how many 10-minute "
      "steps it lasts). Think like a real person -- activities take real time:\n"
      "- Sleeping: don't plan this, the system handles it automatically\n"
      "- Shower/getting dressed: 2-3 steps (20-30 min)\n"
      "- Breakfast/coffee: 2-3 steps\n"
      "- Commuting: 2-4 steps\n"
      "- Focused work (report, analysis): 4-8 steps (40-80 min)\n"
      "- Meeting: 3-6 steps\n"
      "- Lunch/dinner: 3-5 steps\n"
      "- Exercise/jog: 3-5 steps\n"
      "- Movie/TV show: 9-15 steps (1.5-2.5 hours)\n"
      "- Shopping/errands: 3-5 steps\n"
      "- Reading: 3-6 steps\n"
      "- Quick break/coffee: 1-2 steps\n\n"
      "RULES:\n"
      "1. Look at recent_actions_today to see what you already did. "
      "Do NOT repeat the same type of activity you just finished.\n"
      "2. Look at people_talked_to_today. Do NOT initiate conversation with someone "
      "you already talked to today unless you have a genuinely NEW topic. "
      "Most people talk to the same colleague once, maybe twice per day.\n"
      "3. If you want to talk to someone, set interact_with and interact_message. "
      "The conversation will continue naturally for multiple turns.\n"
      "4. Be specific: not 'prepare for work' but 'take a shower and get dressed'.\n"
      "5. If recent_macro_events show economic changes, react to them.\n\n"
      "Return JSON with keys: target_place, action_summary, duration_steps, "
      "pronunciatio, interact_with, interact_message, topic, "
      "spending_multiplier, loan_stress_delta, fraud_susceptibility_delta.\n"
      "Constraints: target_place in [home,work,spending]. "
      "duration_steps is an integer (how many 10-min steps this activity lasts). "
      "pronunciatio is a single emoji. "
      "interact_with can be null or one exact person name. "
      "interact_message can be null or one natural sentence to START a conversation. "
      "topic in [daily_life,family,friendship,hobby,health,community,local_news,"
      "budget_planning,shopping,loan_stress,fraud_warning,scam_rumor]. "
      "spending_multiplier in [0.60,1.80], deltas in [-0.35,0.35]."
    )

  def user_prompt(
    self,
    perception: Dict[str, object],
    thought: Dict[str, object],
    mood: Dict[str, float] | None = None,
    plan: Dict[str, object] | None = None,
    relationships: Dict[str, Dict[str, object]] | None = None,
    last_action: Dict[str, object] | None = None,
    action_history: List[Dict[str, object]] | None = None,
    today_interactions: Dict[str, int] | None = None,
  ) -> str:
    time_info = perception.get("time", {})
    hour = int(time_info.get("hour_of_day", 0))
    minute = int(time_info.get("minute_of_hour", 0))
    day_idx = int(time_info.get("day_of_week_index", 0))
    time_display = time_info.get("time_display", f"{hour:02d}:{minute:02d}")

    payload = {
      "current_time": f"{time_display} ({HOUR_LABELS.get(hour, str(hour))}), {DAY_NAMES[day_idx % 7]}",
      "recent_actions_today": action_history or [],
      "people_talked_to_today": today_interactions or {},
      "perception": {
        k: v for k, v in perception.items()
        if k in ("time", "self", "macro", "policy", "prices", "routine_context",
                 "nearby_people", "social_neighbors", "recent_macro_events")
      },
      "thought": thought,
      "mood": mood or {},
      "todays_plan": plan or {},
      "relationship_context": dict(list((relationships or {}).items())[:6]),
    }
    return "What activity do you start now?\n" + json.dumps(payload, indent=2)

  def default_action(self, thought: Dict[str, object], hour: int = 12) -> Dict[str, object]:
    if 0 <= hour <= 5:
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
    if 6 <= hour <= 7:
      return {
        "target_place": "home",
        "action_summary": "getting ready for the day",
        "pronunciatio": "\u2615",
        "interact_with": None,
        "interact_message": None,
        "topic": "daily_life",
        "spending_multiplier": 1.0,
        "loan_stress_delta": 0.0,
        "fraud_susceptibility_delta": 0.0,
      }
    if 8 <= hour <= 16:
      return {
        "target_place": "work",
        "action_summary": "working at the office",
        "pronunciatio": "\U0001f4bc",
        "interact_with": None,
        "interact_message": None,
        "topic": "daily_life",
        "spending_multiplier": 1.0,
        "loan_stress_delta": 0.0,
        "fraud_susceptibility_delta": 0.0,
      }
    if 17 <= hour <= 20:
      return {
        "target_place": "spending",
        "action_summary": "running errands after work",
        "pronunciatio": "\U0001f6cd\ufe0f",
        "interact_with": None,
        "interact_message": None,
        "topic": "daily_life",
        "spending_multiplier": 1.0,
        "loan_stress_delta": 0.0,
        "fraud_susceptibility_delta": 0.0,
      }
    return {
      "target_place": "home",
      "action_summary": "winding down at home",
      "pronunciatio": "\U0001f3e0",
      "interact_with": None,
      "interact_message": None,
      "topic": "daily_life",
      "spending_multiplier": 1.0,
      "loan_stress_delta": 0.0,
      "fraud_susceptibility_delta": 0.0,
    }

  def decide(
    self,
    perception: Dict[str, object],
    thought: Dict[str, object],
    mood: Dict[str, float] | None = None,
    plan: Dict[str, object] | None = None,
    persona_summary: str = "",
    relationships: Dict[str, Dict[str, object]] | None = None,
    last_action: Dict[str, object] | None = None,
    action_history: List[Dict[str, object]] | None = None,
    today_interactions: Dict[str, int] | None = None,
  ) -> Dict[str, object]:
    hour = int(perception.get("time", {}).get("hour_of_day", 12))
    result = self.client.generate_json(
      self.system_prompt(persona_summary),
      self.user_prompt(perception, thought, mood, plan, relationships,
                       last_action, action_history, today_interactions),
    )
    if not result.get("ok"):
      return {
        "ok": False,
        "error": result.get("error", "action_generation_failed"),
        "action": self.default_action(thought, hour),
        "prompt_tokens": 0,
        "completion_tokens": 0,
      }

    payload = result.get("payload", {})

    target_place = str(payload.get("target_place", "home")).strip().lower()
    if target_place not in {"home", "work", "spending"}:
      target_place = "home"

    topic = str(payload.get("topic", "daily_life")).strip().lower()
    if topic not in ALLOWED_TOPICS:
      topic = "daily_life"

    def _f(value, default=0.0):
      try:
        return float(value)
      except Exception:
        return default

    spending_multiplier = max(0.60, min(1.80, _f(payload.get("spending_multiplier", 1.0), 1.0)))
    loan_stress_delta = max(-0.35, min(0.35, _f(payload.get("loan_stress_delta", 0.0), 0.0)))
    fraud_sus_delta = max(-0.35, min(0.35, _f(payload.get("fraud_susceptibility_delta", 0.0), 0.0)))

    interact_with = payload.get("interact_with", None)
    if interact_with is not None:
      interact_with = str(interact_with).strip() or None

    interact_message = payload.get("interact_message", None)
    if interact_message is not None:
      interact_message = str(interact_message).strip()[:180] or None

    duration_steps = 0
    try:
      duration_steps = int(payload.get("duration_steps", 0))
    except Exception:
      duration_steps = 0

    action = {
      "target_place": target_place,
      "action_summary": str(payload.get("action_summary", "continuing daily routine")).strip()[:180],
      "duration_steps": max(0, min(18, duration_steps)),
      "pronunciatio": str(payload.get("pronunciatio", "\U0001f642")).strip()[:8] or "\U0001f642",
      "interact_with": interact_with,
      "interact_message": interact_message,
      "topic": topic,
      "spending_multiplier": spending_multiplier,
      "loan_stress_delta": loan_stress_delta,
      "fraud_susceptibility_delta": fraud_sus_delta,
    }

    return {
      "ok": True,
      "action": action,
      "prompt_tokens": int(result.get("prompt_tokens", 0)),
      "completion_tokens": int(result.get("completion_tokens", 0)),
    }


class ReactionModule:
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

  def system_prompt(self) -> str:
    return (
      "You are a town resident in a conversation. "
      "Respond naturally as a real person would -- be friendly, curious, guarded, "
      "or dismissive depending on the topic and your situation. "
      "Have a real conversation: ask follow-up questions, share opinions, "
      "react emotionally. Keep each reply 1-3 sentences. "
      "Return JSON with keys: reply, sentiment, wants_to_continue, "
      "spending_delta, loan_stress_delta, fraud_susceptibility_delta. "
      "sentiment in [positive,neutral,negative]. "
      "wants_to_continue: true if you want to keep talking, false if conversation "
      "is wrapping up or you want to end it. deltas in [-0.30,0.30]."
    )

  def user_prompt(
    self, speaker_name: str, listener_profile: Dict[str, object],
    message: str, topic: str,
    conversation_history: List[Dict[str, str]] | None = None,
  ) -> str:
    payload: Dict[str, object] = {
      "from": speaker_name,
      "to_profile": listener_profile,
      "topic": topic,
    }
    if conversation_history:
      payload["conversation_so_far"] = conversation_history
      payload["latest_message"] = message
    else:
      payload["message"] = message
    return "Input JSON:\n" + json.dumps(payload, indent=2)

  def default_reaction(self) -> Dict[str, object]:
    return {
      "reply": "I see, let me think about that.",
      "sentiment": "neutral",
      "wants_to_continue": False,
      "spending_delta": 0.0,
      "loan_stress_delta": 0.0,
      "fraud_susceptibility_delta": 0.0,
    }

  def react(
    self, speaker_name: str, listener_profile: Dict[str, object],
    message: str, topic: str,
    conversation_history: List[Dict[str, str]] | None = None,
  ) -> Dict[str, object]:
    result = self.client.generate_json(
      self.system_prompt(),
      self.user_prompt(speaker_name, listener_profile, message, topic, conversation_history),
    )
    if not result.get("ok"):
      return {
        "ok": False,
        "error": result.get("error", "reaction_generation_failed"),
        "reaction": self.default_reaction(),
        "prompt_tokens": 0,
        "completion_tokens": 0,
      }

    payload = result.get("payload", {})

    sentiment = str(payload.get("sentiment", "neutral")).strip().lower()
    if sentiment not in {"positive", "neutral", "negative"}:
      sentiment = "neutral"

    def _f(value, default=0.0):
      try:
        return float(value)
      except Exception:
        return default

    reaction = {
      "reply": str(payload.get("reply", "I see, let me think about that.")).strip()[:180],
      "sentiment": sentiment,
      "wants_to_continue": bool(payload.get("wants_to_continue", False)),
      "spending_delta": max(-0.30, min(0.30, _f(payload.get("spending_delta", 0.0), 0.0))),
      "loan_stress_delta": max(-0.30, min(0.30, _f(payload.get("loan_stress_delta", 0.0), 0.0))),
      "fraud_susceptibility_delta": max(-0.30, min(0.30, _f(payload.get("fraud_susceptibility_delta", 0.0), 0.0))),
    }

    return {
      "ok": True,
      "reaction": reaction,
      "prompt_tokens": int(result.get("prompt_tokens", 0)),
      "completion_tokens": int(result.get("completion_tokens", 0)),
    }

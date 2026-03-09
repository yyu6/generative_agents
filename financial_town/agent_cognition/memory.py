"""Memory module with importance scoring, recency decay, and keyword retrieval."""

from __future__ import annotations

import math
from typing import Dict, List


IMPORTANCE_BY_TYPE = {
  "reflection": 9,
  "interaction": 7,
  "plan": 6,
  "life_event": 6,
  "macro_event_reaction": 5,
  "mood_shift": 4,
  "thought": 3,
  "routine": 1,
}


class MemoryModule:
  def __init__(self, max_items: int = 2500, recency_decay: float = 0.995):
    self.max_items = max(200, int(max_items))
    self.recency_decay = max(0.9, min(1.0, recency_decay))

  def ensure_initialized(self, state) -> None:
    if not hasattr(state, "active_agent_memory") or state.active_agent_memory is None:
      state.active_agent_memory = []

  def _score_importance(self, item: Dict[str, object]) -> float:
    mem_type = str(item.get("type", "routine"))
    base = IMPORTANCE_BY_TYPE.get(mem_type, 2)
    explicit = item.get("importance")
    if explicit is not None:
      try:
        base = max(base, int(explicit))
      except (ValueError, TypeError):
        pass
    return float(base)

  def remember(self, state, item: Dict[str, object]) -> None:
    self.ensure_initialized(state)
    if "importance" not in item:
      item["importance"] = self._score_importance(item)
    state.active_agent_memory.append(item)
    if len(state.active_agent_memory) > self.max_items:
      self._evict(state)

  def _evict(self, state) -> None:
    """Keep high-importance items longer; drop lowest-scored old items."""
    scored: List[tuple] = []
    n = len(state.active_agent_memory)
    for idx, item in enumerate(state.active_agent_memory):
      imp = float(item.get("importance", 2))
      recency = self.recency_decay ** (n - 1 - idx)
      scored.append((imp * recency, idx))
    scored.sort(key=lambda x: x[0])
    drop_count = n - self.max_items
    drop_indices = {s[1] for s in scored[:drop_count]}
    state.active_agent_memory = [
      item for idx, item in enumerate(state.active_agent_memory)
      if idx not in drop_indices
    ]

  def recent(self, state, limit: int = 12) -> List[Dict[str, object]]:
    self.ensure_initialized(state)
    return state.active_agent_memory[-max(1, int(limit)):]

  def relevant_by_keywords(self, state, keywords: List[str], limit: int = 8) -> List[Dict[str, object]]:
    self.ensure_initialized(state)
    if not keywords:
      return self.recent(state, limit=limit)

    lowered = [k.strip().lower() for k in keywords if k and str(k).strip()]
    if not lowered:
      return self.recent(state, limit=limit)

    scored: List[tuple] = []
    n = len(state.active_agent_memory)
    for idx, item in enumerate(state.active_agent_memory):
      txt = " ".join([
        str(item.get("type", "")),
        str(item.get("summary", "")),
        str(item.get("topic", "")),
        str(item.get("counterparty", "")),
      ]).lower()
      keyword_hits = sum(1 for kw in lowered if kw in txt)
      if keyword_hits == 0:
        continue
      imp = float(item.get("importance", 2))
      recency = self.recency_decay ** (n - 1 - idx)
      score = keyword_hits * imp * recency
      scored.append((score, idx, item))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [s[2] for s in scored[:limit]]

  def important_recent(self, state, limit: int = 10, min_importance: int = 5) -> List[Dict[str, object]]:
    """Retrieve recent high-importance items (reflections, interactions, events)."""
    self.ensure_initialized(state)
    hits: List[Dict[str, object]] = []
    for item in reversed(state.active_agent_memory):
      if float(item.get("importance", 0)) >= min_importance:
        hits.append(item)
      if len(hits) >= limit:
        break
    hits.reverse()
    return hits

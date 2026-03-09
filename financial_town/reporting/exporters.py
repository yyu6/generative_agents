"""State export utilities (JSON + CSV + Stanford-compatible storage)."""

from __future__ import annotations

import csv
import datetime
import json
import shutil
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional

from financial_town.config import SimulationConfig
from financial_town.simulation.state import TownState


def _to_primitive(value: Any) -> Any:
  if isinstance(value, Enum):
    return value.value
  if is_dataclass(value):
    return {k: _to_primitive(v) for k, v in asdict(value).items()}
  if isinstance(value, dict):
    return {k: _to_primitive(v) for k, v in value.items()}
  if isinstance(value, list):
    return [_to_primitive(v) for v in value]
  return value


class SimulationExporter:
  def __init__(self, config: SimulationConfig):
    self.config = config
    self.base_dir = Path(config.output_dir)
    self.snapshots_dir = self.base_dir / "snapshots"
    self.keep_step_snapshots = config.keep_step_snapshots
    self.base_dir.mkdir(parents=True, exist_ok=True)
    self.snapshots_dir.mkdir(parents=True, exist_ok=True)

    self.stanford_enabled = bool(config.export_stanford_storage)
    self.stanford_sim_code = ""
    self.stanford_storage_dir: Optional[Path] = None
    self.stanford_personas_dir: Optional[Path] = None
    self.stanford_movement_dir: Optional[Path] = None
    self.stanford_environment_dir: Optional[Path] = None
    self.stanford_reverie_dir: Optional[Path] = None
    self.frontend_temp_dir: Optional[Path] = None

    if self.stanford_enabled:
      self._init_stanford_export_paths()

  def get_stanford_sim_code(self) -> str:
    return self.stanford_sim_code

  def _init_stanford_export_paths(self) -> None:
    repo_root = Path(__file__).resolve().parents[2]
    frontend_root = repo_root / "environment" / "frontend_server"
    storage_root = frontend_root / "storage"
    temp_root = frontend_root / "temp_storage"

    requested_code = (self.config.stanford_sim_code or "").strip()
    if requested_code:
      self.stanford_sim_code = requested_code
    else:
      ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
      self.stanford_sim_code = f"financial_town_{ts}_{self.config.seed}"

    sim_dir = storage_root / self.stanford_sim_code
    personas_dir = sim_dir / "personas"
    movement_dir = sim_dir / "movement"
    environment_dir = sim_dir / "environment"
    reverie_dir = sim_dir / "reverie"

    personas_dir.mkdir(parents=True, exist_ok=True)
    movement_dir.mkdir(parents=True, exist_ok=True)
    environment_dir.mkdir(parents=True, exist_ok=True)
    reverie_dir.mkdir(parents=True, exist_ok=True)
    temp_root.mkdir(parents=True, exist_ok=True)

    # Clear stale files when reusing the same sim_code.
    for folder in [movement_dir, environment_dir]:
      for file_path in folder.glob("*.json"):
        file_path.unlink()

    self.stanford_storage_dir = sim_dir
    self.stanford_personas_dir = personas_dir
    self.stanford_movement_dir = movement_dir
    self.stanford_environment_dir = environment_dir
    self.stanford_reverie_dir = reverie_dir
    self.frontend_temp_dir = temp_root

  def _sim_datetime(self, step: int) -> datetime.datetime:
    try:
      base = datetime.datetime.strptime(self.config.stanford_start_date, "%B %d, %Y")
    except Exception:
      base = datetime.datetime(2023, 2, 13)
    return base + datetime.timedelta(seconds=self.config.stanford_sec_per_step * step)

  def _sim_time_str(self, step: int) -> str:
    return self._sim_datetime(step).strftime("%B %d, %Y, %H:%M:%S")

  def _split_name(self, full_name: str):
    parts = full_name.split(" ")
    if not parts:
      return "", ""
    if len(parts) == 1:
      return parts[0], ""
    return " ".join(parts[:-1]), parts[-1]

  def _minimal_scratch(self, person, step: int) -> Dict[str, Any]:
    first_name, last_name = self._split_name(person.full_name)
    now_str = self._sim_time_str(step)
    return {
      "first_name": first_name,
      "last_name": last_name,
      "age": person.age,
      "curr_time": now_str,
      "curr_tile": [person.current_x, person.current_y],
      "vision_r": 4,
      "att_bandwidth": 3,
      "retention": 10,
      "innate": f"{person.spending_style} spender",
      "learned": f"{person.industry} worker",
      "currently": f"{person.occupation} in {person.current_place}",
      "lifestyle": "structured daily routine",
      "daily_req": [
        "Commute between home, work, and spending area",
        "Manage budget, debt, and household obligations",
      ],
      "f_daily_schedule": [
        ["home", 7],
        ["work", 9],
        ["spending", 3],
        ["home", 5],
      ],
      "act_address": person.home_address,
      "act_start_time": now_str,
      "act_duration": 60,
      "act_description": "idle",
      "act_pronunciatio": "🙂",
    }

  def _ensure_minimal_bootstrap(self, persona_dir: Path, person, step: int) -> None:
    bootstrap = persona_dir / "bootstrap_memory"
    assoc = bootstrap / "associative_memory"
    assoc.mkdir(parents=True, exist_ok=True)

    scratch_path = bootstrap / "scratch.json"
    if not scratch_path.exists():
      scratch_path.write_text(
        json.dumps(self._minimal_scratch(person, step), indent=2),
        encoding="utf-8",
      )

    spatial_path = bootstrap / "spatial_memory.json"
    if not spatial_path.exists():
      spatial_path.write_text("{}", encoding="utf-8")

    nodes_path = assoc / "nodes.json"
    if not nodes_path.exists():
      nodes_path.write_text("{}", encoding="utf-8")

  def _refresh_scratch_runtime(self, persona_dir: Path, person, step: int) -> None:
    scratch_path = persona_dir / "bootstrap_memory" / "scratch.json"
    try:
      if scratch_path.exists():
        scratch = json.loads(scratch_path.read_text(encoding="utf-8"))
      else:
        scratch = self._minimal_scratch(person, step)
    except Exception:
      scratch = self._minimal_scratch(person, step)

    now_str = self._sim_time_str(step)
    scratch["curr_time"] = now_str
    scratch["curr_tile"] = [person.current_x, person.current_y]
    scratch["act_start_time"] = now_str
    scratch["act_address"] = {
      "home": person.home_address,
      "work": person.work_address,
      "spending": person.spending_address,
    }.get(person.current_place, person.home_address)
    scratch["act_description"] = f"{person.current_place} routine"

    scratch_path.write_text(json.dumps(scratch, indent=2), encoding="utf-8")

  def _ensure_persona_dirs(self, state: TownState) -> None:
    if not self.stanford_personas_dir or not self.stanford_storage_dir:
      return

    fork_personas_dir = (
      self.stanford_storage_dir.parent
      / self.config.stanford_fork_sim_code
      / "personas"
    )

    for person in state.people.values():
      persona_dir = self.stanford_personas_dir / person.full_name
      if not persona_dir.exists() and fork_personas_dir.exists():
        source_dir = fork_personas_dir / person.full_name
        if source_dir.exists():
          shutil.copytree(source_dir, persona_dir)

      self._ensure_minimal_bootstrap(persona_dir, person, state.step)
      self._refresh_scratch_runtime(persona_dir, person, state.step)

  def _movement_payload_for_person(self, state: TownState, person_id: str) -> Dict[str, Any]:
    person = state.people[person_id]
    movement = state.step_movements.get(person_id, {})

    coords = movement.get("movement", [person.current_x, person.current_y])
    description = movement.get(
      "description",
      f"idle @ {person.current_place}",
    )
    pronunciatio = movement.get("pronunciatio", "🙂")

    chat = movement.get("chat")
    if isinstance(chat, str):
      chat = [[person.full_name, chat]]
    if not isinstance(chat, list):
      chat = None

    return {
      "movement": [int(coords[0]), int(coords[1])],
      "pronunciatio": pronunciatio,
      "description": description,
      "chat": chat,
    }

  def _write_reverie_meta(self, state: TownState) -> None:
    if not self.stanford_reverie_dir:
      return

    meta = {
      "fork_sim_code": self.config.stanford_fork_sim_code,
      "start_date": self.config.stanford_start_date,
      "curr_time": self._sim_time_str(state.step),
      "sec_per_step": self.config.stanford_sec_per_step,
      "maze_name": "the_ville",
      "persona_names": [p.full_name for p in state.people.values()],
      "step": state.step,
    }
    meta_path = self.stanford_reverie_dir / "meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

  def _write_stanford_step_files(self, state: TownState) -> None:
    if not self.stanford_enabled:
      return
    if not self.stanford_movement_dir or not self.stanford_environment_dir:
      return

    self._ensure_persona_dirs(state)

    movement_payload = {
      "persona": {},
      "meta": {"curr_time": self._sim_time_str(state.step)},
    }
    environment_payload = {}

    for person in state.people.values():
      movement_payload["persona"][person.full_name] = self._movement_payload_for_person(
        state,
        person.person_id,
      )
      environment_payload[person.full_name] = {
        "maze": "the_ville",
        "x": int(person.current_x),
        "y": int(person.current_y),
      }

    movement_path = self.stanford_movement_dir / f"{state.step}.json"
    movement_path.write_text(json.dumps(movement_payload, indent=2), encoding="utf-8")

    environment_path = self.stanford_environment_dir / f"{state.step}.json"
    environment_path.write_text(json.dumps(environment_payload, indent=2), encoding="utf-8")

    self._write_agent_state(state)
    self._write_reverie_meta(state)

  def _write_agent_state(self, state: TownState) -> None:
    if not self.stanford_storage_dir:
      return
    agent_state_dir = self.stanford_storage_dir / "agent_state"
    agent_state_dir.mkdir(parents=True, exist_ok=True)

    step_memories = [
      m for m in state.active_agent_memory
      if m.get("step") == state.step
    ]
    step_thoughts = [
      t for t in state.active_agent_thoughts
      if t.get("step") == state.step
    ]

    payload = {
      "step": state.step,
      "time": self._sim_time_str(state.step),
      "mood": _to_primitive(getattr(state, "active_agent_mood", {})),
      "plan": _to_primitive(getattr(state, "active_agent_plan", {})),
      "relationships": _to_primitive(getattr(state, "active_agent_relationships", {})),
      "step_memories": _to_primitive(step_memories),
      "step_thoughts": _to_primitive(step_thoughts),
      "total_memory_count": len(state.active_agent_memory),
      "recent_memories": _to_primitive(state.active_agent_memory[-8:]),
      "recent_reflections": _to_primitive(getattr(state, "active_agent_reflections", [])[-3:]),
      "recent_macro_events": _to_primitive(state.macro_events[-4:]),
    }

    path = agent_state_dir / f"{state.step}.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

  def _update_frontend_temp_files(self, start_step: int = 0) -> None:
    if not self.frontend_temp_dir:
      return
    sim_code_payload = {"sim_code": self.stanford_sim_code}
    step_payload = {"step": int(start_step)}

    (self.frontend_temp_dir / "curr_sim_code.json").write_text(
      json.dumps(sim_code_payload, indent=2),
      encoding="utf-8",
    )
    (self.frontend_temp_dir / "curr_step.json").write_text(
      json.dumps(step_payload, indent=2),
      encoding="utf-8",
    )

  def export_step_snapshot(self, state: TownState) -> None:
    if self.keep_step_snapshots:
      payload = {
        "step": state.step,
        "scenario_name": state.scenario_name,
        "policy": _to_primitive(state.policy),
        "macro": _to_primitive(state.macro),
        "prices": {
          "consumer_price_index": float(getattr(state, "consumer_price_index", 1.0) or 1.0),
          "category_price_multipliers": _to_primitive(getattr(state, "category_price_multipliers", {}) or {}),
        },
        "households": _to_primitive(state.households),
        "people": _to_primitive(state.people),
        "properties": _to_primitive(state.properties),
        "loans": _to_primitive(state.loans),
        "metrics": state.metrics_history[-1] if state.metrics_history else {},
        "interactions_step": [
          event for event in state.interaction_events if event.get("step") == state.step
        ],
        "active_agent": {
          "active_agent_id": state.active_agent_id,
          "memory_size": len(state.active_agent_memory),
          "thoughts_size": len(state.active_agent_thoughts),
          "recent_memory": _to_primitive(state.active_agent_memory[-12:]),
          "recent_thoughts": _to_primitive(state.active_agent_thoughts[-12:]),
        },
        "macro_events_recent": _to_primitive(state.macro_events[-12:]),
        "llm_step": {
          "stats": _to_primitive(state.llm_stats),
          "events": [
            event for event in state.llm_events if event.get("step") == state.step
          ],
        },
      }
      path = self.snapshots_dir / f"step_{state.step:04d}.json"
      path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    self._write_stanford_step_files(state)

  def _export_metrics_csv(self, state: TownState) -> None:
    if not state.metrics_history:
      return
    path = self.base_dir / "metrics.csv"
    keys = list(state.metrics_history[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
      writer = csv.DictWriter(f, fieldnames=keys)
      writer.writeheader()
      writer.writerows(state.metrics_history)

  def _export_transactions_csv(self, state: TownState) -> None:
    path = self.base_dir / "transactions.csv"
    rows = [_to_primitive(tx) for tx in state.transactions]
    if not rows:
      path.write_text("", encoding="utf-8")
      return
    keys = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
      writer = csv.DictWriter(f, fieldnames=keys)
      writer.writeheader()
      writer.writerows(rows)

  def _export_final_state_json(self, state: TownState) -> None:
    payload = {
      "step": state.step,
      "scenario_name": state.scenario_name,
      "policy": _to_primitive(state.policy),
      "macro": _to_primitive(state.macro),
      "prices": {
        "consumer_price_index": float(getattr(state, "consumer_price_index", 1.0) or 1.0),
        "category_price_multipliers": _to_primitive(getattr(state, "category_price_multipliers", {}) or {}),
      },
      "metrics_final": state.metrics_history[-1] if state.metrics_history else {},
      "counts": {
        "people": len(state.people),
        "households": len(state.households),
        "properties": len(state.properties),
        "loans": len(state.loans),
        "transactions": len(state.transactions),
        "interaction_events": len(state.interaction_events),
        "llm_events": len(state.llm_events),
        "macro_events": len(state.macro_events),
        "active_agent_memory": len(state.active_agent_memory),
      },
      "llm_stats": _to_primitive(state.llm_stats),
      "active_agent": {
        "active_agent_id": state.active_agent_id,
        "scripted_population_size": len(state.scripted_agent_ids),
        "recent_thoughts": _to_primitive(state.active_agent_thoughts[-40:]),
      },
      "macro_events": _to_primitive(state.macro_events),
      "stanford_storage": {
        "enabled": self.stanford_enabled,
        "sim_code": self.stanford_sim_code,
        "storage_dir": str(self.stanford_storage_dir) if self.stanford_storage_dir else "",
      },
    }
    path = self.base_dir / "final_summary.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

  def export_final(self, state: TownState) -> None:
    self._export_metrics_csv(state)
    self._export_transactions_csv(state)
    self._export_final_state_json(state)

    if self.stanford_enabled:
      self._write_reverie_meta(state)
      if self.config.update_temp_files_for_simulator_home:
        self._update_frontend_temp_files(start_step=0)

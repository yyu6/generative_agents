"""Configuration models for Financial Town."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


@dataclass
class SimulationConfig:
  """Top-level runtime configuration for the simulator."""

  seed: int = 42
  households: int = 120
  max_household_size: int = 5
  home_ownership_rate: float = 0.62

  baseline_policy_path: str = "financial_town/data/policies/base_policy.json"
  scenario_policy_path: str = "financial_town/data/policies/baseline_scenario.json"

  fraud_transaction_rate: float = 0.012
  fraudulent_loan_rate: float = 0.004

  output_dir: str = "financial_town/output"
  export_every_step: bool = True
  keep_step_snapshots: bool = True

  # Behavior profile options:
  #   "synthetic"   -> generated population
  #   "stanford_n3" -> Isabella/Maria/Klaus test profile
  #   "single_agent_world" -> 1 free agent + scripted population
  population_profile: str = "synthetic"
  steps_per_day: int = 144
  economy_interval_steps: int = 1
  scripted_population_size: int = 100
  active_agent_name: str = "Alex Carter"

  # Toggle behavior layers.
  enable_social_interaction: bool = True
  enable_mobility: bool = True

  # LLM-agent mode (uses API-backed agent decisions instead of pure rules).
  enable_llm_agents: bool = True
  llm_provider: str = "openai"  # "gemini" or "openai"
  llm_model: str = "gpt-4o-mini"
  llm_temperature: float = 0.4
  llm_max_output_tokens: int = 280
  llm_timeout_sec: int = 25
  llm_max_agents_per_step: int = 12
  llm_disable_rule_social: bool = True
  llm_strict_mode: bool = True
  llm_log_prompts: bool = False

  # LLM-Economist style macro event generation.
  enable_llm_economist: bool = True
  llm_economist_provider: str = "openai"
  llm_economist_model: str = "gpt-4o-mini"
  llm_economist_temperature: float = 0.25
  llm_economist_max_output_tokens: int = 360
  llm_economist_every_steps: int = 24

  # Stanford-compatible storage export (same folder schema as Generative Town).
  export_stanford_storage: bool = False
  stanford_sim_code: str = ""
  stanford_fork_sim_code: str = "base_the_ville_isabella_maria_klaus"
  stanford_start_date: str = "February 13, 2023"
  stanford_sec_per_step: int = 600
  update_temp_files_for_simulator_home: bool = True

  def _load_json(self, path_value: str) -> Dict[str, Any]:
    path = Path(path_value)
    if not path.exists():
      raise FileNotFoundError(f"Policy file does not exist: {path.resolve()}")
    with path.open("r", encoding="utf-8") as f:
      return json.load(f)

  def load_policy_bundle(self) -> Dict[str, Any]:
    """
    Merge base policy and scenario policy into one runtime bundle.

    Expected file contract:
      base policy file:
        {
          "policy": {...},
          "macro": {...}
        }
      scenario file:
        {
          "name": "scenario_name",
          "initial_policy_overrides": {...},
          "initial_macro_overrides": {...},
          "events": [{"step": 6, "policy": {...}, "macro": {...}, "note": "..."}]
        }
    """
    base_payload = self._load_json(self.baseline_policy_path)
    scenario_payload = self._load_json(self.scenario_policy_path)

    policy = dict(base_payload.get("policy", {}))
    macro = dict(base_payload.get("macro", {}))

    policy.update(scenario_payload.get("initial_policy_overrides", {}))
    macro.update(scenario_payload.get("initial_macro_overrides", {}))

    return {
      "scenario_name": scenario_payload.get("name", "custom"),
      "policy": policy,
      "macro": macro,
      "events": scenario_payload.get("events", []),
    }

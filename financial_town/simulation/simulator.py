"""Main orchestrator for Financial Town."""

from __future__ import annotations

import random

from financial_town.config import SimulationConfig
from financial_town.engines import (
  FinanceEngine,
  FraudEngine,
  HousingEngine,
  LaborEngine,
  LLMEconomistEngine,
  LLMAgentEngine,
  MobilityEngine,
  PolicyEngine,
  SingleAgentEngine,
  SocialEngine,
)
from financial_town.generators import build_initial_town, build_single_agent_world, build_stanford_n3_town
from financial_town.reporting import MetricsEngine, SimulationExporter
from financial_town.simulation.state import TownState


class FinancialTownSimulator:
  """Coordinates all engines and manages per-step execution."""

  def __init__(self, config: SimulationConfig):
    self.config = config
    self.rng = random.Random(config.seed)
    bundle = config.load_policy_bundle()

    if config.population_profile == "stanford_n3":
      self.state = build_stanford_n3_town(policy_bundle=bundle)
    elif config.population_profile == "single_agent_world":
      self.state = build_single_agent_world(config=config, policy_bundle=bundle, rng=self.rng)
    else:
      self.state = build_initial_town(config=config, policy_bundle=bundle, rng=self.rng)
    self.state.scenario_name = bundle.get("scenario_name", "custom")

    self.policy_engine = PolicyEngine(bundle.get("events", []), self.rng)
    self.labor_engine = LaborEngine(self.rng)
    self.finance_engine = FinanceEngine(self.rng)
    self.housing_engine = HousingEngine(self.rng)
    self.fraud_engine = FraudEngine(self.rng, config)
    self.mobility_engine = MobilityEngine(self.rng, steps_per_day=config.steps_per_day)
    self.social_engine = SocialEngine(self.rng)
    self.llm_agent_engine = LLMAgentEngine(config, self.rng) if config.enable_llm_agents else None
    self.single_agent_engine = (
      SingleAgentEngine(config, self.rng)
      if config.enable_llm_agents and config.population_profile == "single_agent_world"
      else None
    )
    self.llm_economist_engine = LLMEconomistEngine(config, self.rng) if config.enable_llm_economist else None
    self.metrics_engine = MetricsEngine()
    self.exporter = SimulationExporter(config)

  def _run_economy_step(self) -> bool:
    interval = max(1, self.config.economy_interval_steps)
    return self.state.step % interval == 0

  def run(self, steps: int) -> TownState:
    if not self.config.enable_llm_agents:
      raise RuntimeError("LLM-only mode is enabled. Set enable_llm_agents=True.")

    for _ in range(steps):
      if self.config.population_profile == "single_agent_world":
        self.state.step_movements = {}
        self.state.step_chat = {}
        if not self.single_agent_engine:
          raise RuntimeError("SingleAgentEngine is not initialized.")
        self.single_agent_engine.run_step(self.state)
      else:
        # LLM-only mode for legacy profiles.
        self.state.step_movements = {}
        self.state.step_chat = {}
        if not self.llm_agent_engine:
          raise RuntimeError("LLM engine is not initialized.")
        self.llm_agent_engine.run_step(self.state)

      if self._run_economy_step():
        if self.llm_economist_engine:
          self.llm_economist_engine.run_step(self.state)
        self.policy_engine.run_step(self.state)
        self.labor_engine.run_step(self.state)
        self.finance_engine.run_step(self.state)
        self.housing_engine.run_step(self.state)
        self.fraud_engine.run_step(self.state)

      metrics = self.metrics_engine.compute_step_metrics(self.state)
      self.state.metrics_history.append(metrics)

      if self.config.export_every_step or self.config.export_stanford_storage:
        self.exporter.export_step_snapshot(self.state)

      self.state.step += 1

    self.exporter.export_final(self.state)
    return self.state

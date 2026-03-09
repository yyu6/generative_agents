"""CLI entrypoint for running Financial Town simulations."""

from __future__ import annotations

import argparse

from financial_town import FinancialTownSimulator, SimulationConfig

SCENARIO_MAP = {
  "baseline": "financial_town/data/policies/baseline_scenario.json",
  "recession": "financial_town/data/policies/recession_scenario.json",
  "rate_hike": "financial_town/data/policies/rate_hike_scenario.json",
}


def build_parser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(description="Run a Financial Town simulation.")
  parser.add_argument("--steps", type=int, default=48, help="Number of steps to simulate.")
  parser.add_argument("--households", type=int, default=120, help="Number of households in the town.")
  parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
  parser.add_argument(
    "--profile",
    choices=["synthetic", "stanford_n3", "single_agent_world"],
    default="single_agent_world",
    help="Population profile: synthetic population or Stanford n3 personas.",
  )
  parser.add_argument(
    "--scripted-population-size",
    type=int,
    default=100,
    help="Population size for scripted residents in single_agent_world mode.",
  )
  parser.add_argument(
    "--active-agent-name",
    type=str,
    default="Alex Carter",
    help="Name of the single free agent in single_agent_world mode.",
  )
  parser.add_argument(
    "--scenario",
    choices=sorted(SCENARIO_MAP.keys()),
    default="baseline",
    help="Policy/macroeconomic scenario profile.",
  )
  parser.add_argument(
    "--scenario-file",
    type=str,
    default="",
    help="Optional path to a custom scenario JSON file. Overrides --scenario.",
  )
  parser.add_argument(
    "--output-dir",
    type=str,
    default="financial_town/output",
    help="Output directory for snapshots and reports.",
  )
  parser.add_argument(
    "--economy-interval-steps",
    type=int,
    default=1,
    help="Run economy engines every N steps (useful with fine-grained mobility steps).",
  )
  parser.add_argument(
    "--steps-per-day",
    type=int,
    default=144,
    help="Steps per simulated day (144 = 10-minute steps, 24 = 1-hour steps).",
  )
  parser.add_argument(
    "--llm-agents",
    action="store_true",
    default=True,
    help="Enable LLM-driven agent decisions.",
  )
  parser.add_argument(
    "--llm-provider",
    choices=["gemini", "openai"],
    default="openai",
    help="LLM provider for agent decisions.",
  )
  parser.add_argument(
    "--llm-model",
    type=str,
    default="",
    help="LLM model name (provider-specific).",
  )
  parser.add_argument(
    "--llm-max-agents-per-step",
    type=int,
    default=1,
    help="Max number of agent LLM calls per step.",
  )
  parser.add_argument(
    "--llm-temperature",
    type=float,
    default=0.4,
    help="LLM sampling temperature.",
  )
  parser.add_argument(
    "--llm-strict",
    action="store_true",
    default=True,
    help="Fail run if LLM mode is enabled but API call fails.",
  )
  parser.add_argument(
    "--llm-allow-rule-social",
    action="store_true",
    help="Keep rule-based social engine enabled alongside LLM mode.",
  )
  parser.add_argument(
    "--disable-llm-economist",
    action="store_true",
    help="Disable LLM-Economist style macro event generation.",
  )
  parser.add_argument(
    "--llm-economist-every-steps",
    type=int,
    default=24,
    help="Generate LLM-Economist macro events every N steps.",
  )
  parser.add_argument(
    "--stanford-storage",
    action="store_true",
    help="Export to Stanford-compatible storage folder under environment/frontend_server/storage.",
  )
  parser.add_argument(
    "--sim-code",
    type=str,
    default="",
    help="Target simulation code when --stanford-storage is enabled.",
  )
  parser.add_argument(
    "--fork-sim-code",
    type=str,
    default="base_the_ville_isabella_maria_klaus",
    help="Fork simulation code used as persona bootstrap source in Stanford export mode.",
  )
  parser.add_argument(
    "--no-step-export",
    action="store_true",
    help="Disable per-step snapshot JSON export.",
  )
  return parser


def main() -> None:
  args = build_parser().parse_args()
  scenario_path = args.scenario_file or SCENARIO_MAP[args.scenario]

  config = SimulationConfig(
    seed=args.seed,
    households=args.households,
    scenario_policy_path=scenario_path,
    output_dir=args.output_dir,
    export_every_step=not args.no_step_export,
    population_profile=args.profile,
    steps_per_day=max(4, int(args.steps_per_day)),
    economy_interval_steps=max(1, int(args.economy_interval_steps)),
    scripted_population_size=max(1, int(args.scripted_population_size)),
    active_agent_name=args.active_agent_name.strip() or "Alex Carter",
    enable_llm_agents=True,
    llm_provider=args.llm_provider,
    llm_model=(args.llm_model.strip() or ("gpt-4o-mini" if args.llm_provider == "openai" else "gemini-1.5-flash")),
    llm_temperature=float(args.llm_temperature),
    llm_max_agents_per_step=max(1, int(args.llm_max_agents_per_step)),
    llm_disable_rule_social=True,
    llm_strict_mode=True,
    enable_llm_economist=not args.disable_llm_economist,
    llm_economist_provider=args.llm_provider,
    llm_economist_model=(args.llm_model.strip() or ("gpt-4o-mini" if args.llm_provider == "openai" else "gemini-1.5-flash")),
    llm_economist_every_steps=max(1, int(args.llm_economist_every_steps)),
    export_stanford_storage=args.stanford_storage,
    stanford_sim_code=args.sim_code.strip(),
    stanford_fork_sim_code=args.fork_sim_code.strip(),
    stanford_sec_per_step=86400 // max(4, int(args.steps_per_day)),
  )
  simulator = FinancialTownSimulator(config)
  final_state = simulator.run(args.steps)

  if final_state.metrics_history:
    final = final_state.metrics_history[-1]
    print("Financial Town simulation complete.")
    print(f"Scenario: {final['scenario']}")
    print(f"Step: {final['step']}")
    print(f"Population: {final['population']}")
    print(f"Employed rate: {final['employed_rate']:.2%}")
    print(f"Total debt: ${final['total_debt']:,.2f}")
    print(f"Fraudulent txns (step): {final['fraudulent_txn_step']}")
    print(f"Fraudulent txns (total): {final['fraudulent_txn_total']}")
    print(f"Interactions (step): {final.get('interaction_events_step', 0)}")
    print(f"LLM calls (step): {final.get('llm_calls_step', 0)}")
    print(f"LLM calls (total): {final.get('llm_calls_total', 0)}")
    print(f"LLM errors (total): {final.get('llm_errors_total', 0)}")
    if final.get("llm_prompt_tokens_total") or final.get("llm_completion_tokens_total"):
      print(
        "LLM tokens (prompt/completion total): "
        f"{final.get('llm_prompt_tokens_total', 0)}/"
        f"{final.get('llm_completion_tokens_total', 0)}"
      )
  print(f"Output directory: {config.output_dir}")
  if args.stanford_storage:
    print(f"Stanford sim_code: {simulator.exporter.get_stanford_sim_code()}")


if __name__ == "__main__":
  main()

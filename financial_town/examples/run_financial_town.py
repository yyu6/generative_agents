"""Example runner for quick experiments."""

from financial_town import FinancialTownSimulator, SimulationConfig


def main() -> None:
  config = SimulationConfig(
    seed=17,
    households=80,
    scenario_policy_path="financial_town/data/policies/recession_scenario.json",
    output_dir="financial_town/output/example_recession",
  )
  sim = FinancialTownSimulator(config)
  state = sim.run(steps=12)
  print(state.metrics_history[-1])


if __name__ == "__main__":
  main()

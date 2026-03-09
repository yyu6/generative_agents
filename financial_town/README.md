# Financial Town

Financial Town is a structured, extensible simulator for:

1. One free LLM agent with explicit `perception -> memory -> thought -> action` modules
2. Scripted resident population (e.g., 100 fixed profiles in a Truman-world style setting)
3. LLM-generated resident reactions/feedback when the free agent initiates interactions
4. Household cashflow and debt dynamics
5. LLM-Economist style macro event generation
6. Stanford-compatible storage export (`environment/frontend_server/storage/<sim_code>/...`)

## Directory Layout

```text
financial_town/
  cli.py
  config.py
  models/
    enums.py
    entities.py
  generators/
    people_generator.py
    stanford_n3_generator.py
    single_agent_world_generator.py
  agent_cognition/
    perception.py
    memory.py
    thought.py
    action.py
  engines/
    policy_engine.py
    labor_engine.py
    finance_engine.py
    housing_engine.py
    fraud_engine.py
    llm_economist_engine.py
    single_agent_engine.py
    llm_agent_engine.py
  llm/
    client.py
  simulation/
    state.py
    simulator.py
  reporting/
    metrics.py
    exporters.py
  data/policies/
    base_policy.json
    baseline_scenario.json
    recession_scenario.json
    rate_hike_scenario.json
```

## Quick Start (Single-Agent World, OpenAI)

From repo root:

```bash
source .venv/bin/activate
export OPENAI_API_KEY=\"...\"
python -m financial_town.cli \
  --profile single_agent_world \
  --scripted-population-size 100 \
  --active-agent-name \"Alex Carter\" \
  --scenario baseline \
  --steps 72 \
  --steps-per-day 24 \
  --economy-interval-steps 24 \
  --llm-provider openai \
  --llm-model gpt-4o-mini \
  --llm-max-agents-per-step 1 \
  --llm-economist-every-steps 24 \
  --stanford-storage \
  --sim-code test-simulation \
  --fork-sim-code base_the_ville_isabella_maria_klaus
```

## LLM Agent Mode (Gemini/OpenAI)

Set key in your shell:

```bash
export GOOGLE_API_KEY="..."
# or
export OPENAI_API_KEY="..."
```

Run LLM-driven agents:

```bash
python -m financial_town.cli \
  --profile single_agent_world \
  --scripted-population-size 100 \
  --active-agent-name \"Alex Carter\" \
  --scenario baseline \
  --steps 72 \
  --steps-per-day 24 \
  --economy-interval-steps 24 \
  --llm-agents \
  --llm-provider openai \
  --llm-model gpt-4o-mini \
  --llm-max-agents-per-step 1 \
  --llm-temperature 0.4 \
  --llm-economist-every-steps 24 \
  --stanford-storage \
  --sim-code test-simulation \
  --fork-sim-code base_the_ville_isabella_maria_klaus
```

Note: this project is configured as LLM-only agent mode with strict execution.
If API is unavailable or key is missing, the run fails instead of falling back to rule social logic.

## Run With Stanford n3 Personas + Stanford Storage Export

```bash
source .venv/bin/activate
python -m financial_town.cli \
  --profile stanford_n3 \
  --scenario baseline \
  --steps 72 \
  --steps-per-day 24 \
  --economy-interval-steps 24 \
  --stanford-storage \
  --sim-code test-simulation \
  --fork-sim-code base_the_ville_isabella_maria_klaus
```

This writes files to:

- `environment/frontend_server/storage/test-simulation/movement/*.json`
- `environment/frontend_server/storage/test-simulation/environment/*.json`
- `environment/frontend_server/storage/test-simulation/reverie/meta.json`
- `environment/frontend_server/storage/test-simulation/personas/*`
- `environment/frontend_server/temp_storage/curr_sim_code.json`
- `environment/frontend_server/temp_storage/curr_step.json`

Then open:

- `http://127.0.0.1:8000/financial_town` to run from UI
- `http://127.0.0.1:8000/replay/test-simulation/0/` to replay
- `http://127.0.0.1:8000/simulator_home` after temp files are refreshed

## Outputs

Default output directory (`financial_town/output/...`) includes:

- `metrics.csv`
- `transactions.csv`
- `final_summary.json`
- `snapshots/step_XXXX.json` (if enabled)

## Notes

- Current runtime is configured as LLM strict mode for the free agent path: missing keys/API failures will fail fast.
- Step/final metrics include LLM call/error/token counters and macro event traces.

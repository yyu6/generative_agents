"""
Author: Joon Sung Park (joonspk@stanford.edu)
File: views.py
"""
import os
import string
import random
import json
import traceback
import sys
from os import listdir
from pathlib import Path

import datetime
from django.shortcuts import render, redirect, HttpResponseRedirect
from django.http import HttpResponse, JsonResponse
from global_methods import *

from django.contrib.staticfiles.templatetags.staticfiles import static
from .models import *

# Allow importing project-level modules when Django runs from frontend_server/.
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
  sys.path.append(str(REPO_ROOT))

from financial_town import FinancialTownSimulator, SimulationConfig


FINANCIAL_TOWN_SCENARIOS = {
  "baseline": "financial_town/data/policies/baseline_scenario.json",
  "recession": "financial_town/data/policies/recession_scenario.json",
  "rate_hike": "financial_town/data/policies/rate_hike_scenario.json",
}

def landing(request): 
  context = {}
  template = "landing/landing.html"
  return render(request, template, context)


def _to_bool(value):
  if isinstance(value, bool):
    return value
  if isinstance(value, str):
    return value.lower() in ["true", "1", "yes", "y"]
  return bool(value)


def _clamp_int(value, lower, upper, default):
  try:
    parsed = int(value)
  except:
    return default
  return max(lower, min(upper, parsed))


def _clamp_float(value, lower, upper, default):
  try:
    parsed = float(value)
  except:
    return default
  return max(lower, min(upper, parsed))


def _safe_float(numerator, denominator):
  if denominator == 0:
    return 0.0
  return numerator / denominator


def _household_risk_rows(state):
  rows = []
  for household in state.households.values():
    debt = 0.0
    liquid = 0.0
    gross_income = 0.0
    for person_id in household.member_ids:
      person = state.people[person_id]
      liquid += person.checking_balance + person.savings_balance
      gross_income += person.annual_base_income / 12.0
      for loan_id in person.loan_ids:
        if loan_id in state.loans:
          debt += state.loans[loan_id].remaining_principal
    rows.append({
      "household_id": household.household_id,
      "city_zone": household.city_zone,
      "housing_mode": household.housing_mode,
      "members": len(household.member_ids),
      "monthly_income": round(gross_income, 2),
      "liquid_cash": round(liquid, 2),
      "debt": round(debt, 2),
      "debt_to_liquid_ratio": round(_safe_float(debt, max(1.0, liquid)), 3),
    })
  rows.sort(key=lambda x: (x["debt_to_liquid_ratio"], x["debt"]), reverse=True)
  return rows[:12]


def _fraud_rows(state):
  rows = []
  fraud_txns = [tx for tx in state.transactions if tx.fraudulent]
  fraud_txns = sorted(fraud_txns, key=lambda tx: tx.step, reverse=True)
  for tx in fraud_txns[:30]:
    rows.append({
      "transaction_id": tx.transaction_id,
      "step": tx.step,
      "person_id": tx.person_id,
      "household_id": tx.household_id,
      "category": tx.category,
      "type": tx.tx_type.value,
      "amount": tx.amount,
      "note": tx.note,
    })
  return rows


def _interaction_rows(state):
  rows = []
  events = sorted(state.interaction_events, key=lambda evt: evt.get("step", -1), reverse=True)
  for evt in events[:40]:
    rows.append({
      "step": evt.get("step"),
      "people": ", ".join(evt.get("people", [])),
      "relations": ", ".join(evt.get("relations", [])),
      "topic": evt.get("topic", ""),
      "same_place": bool(evt.get("same_place", False)),
      "distance": evt.get("distance", 0),
    })
  return rows


def financial_town_dashboard(request):
  context = {
    "scenario_options": list(FINANCIAL_TOWN_SCENARIOS.keys()),
    "default_households": 101,
    "default_steps": 72,
    "default_steps_per_day": 24,
    "default_economy_interval_steps": 24,
    "default_population_profile": "single_agent_world",
    "default_scripted_population_size": 100,
    "default_active_agent_name": "Alex Carter",
    "default_fork_sim_code": "base_the_ville_isabella_maria_klaus",
    "default_enable_llm_agents": True,
    "default_llm_provider": "openai",
    "default_llm_model": "gpt-4o-mini",
    "default_llm_max_agents_per_step": 1,
    "default_llm_temperature": 0.4,
    "default_llm_strict_mode": True,
    "default_enable_llm_economist": True,
    "default_llm_economist_every_steps": 24,
    "default_seed": 42,
  }
  template = "financial_town/dashboard.html"
  return render(request, template, context)


def financial_town_run(request):
  if request.method != "POST":
    return JsonResponse({"ok": False, "error": "Use POST."}, status=405)

  try:
    body = json.loads(request.body or "{}")
    scenario_name = str(body.get("scenario", "baseline")).strip()
    if scenario_name not in FINANCIAL_TOWN_SCENARIOS:
      scenario_name = "baseline"

    steps = _clamp_int(body.get("steps", 72), 1, 240, 72)
    households = _clamp_int(body.get("households", 120), 10, 500, 120)
    seed = _clamp_int(body.get("seed", 42), 1, 99999999, 42)
    keep_snapshots = _to_bool(body.get("keep_snapshots", True))
    population_profile = str(body.get("population_profile", "synthetic")).strip()
    if population_profile not in ["synthetic", "stanford_n3", "single_agent_world"]:
      population_profile = "single_agent_world"
    if population_profile == "stanford_n3":
      households = 3
    scripted_population_size = _clamp_int(body.get("scripted_population_size", 100), 1, 2000, 100)
    active_agent_name = str(body.get("active_agent_name", "Alex Carter")).strip() or "Alex Carter"

    steps_per_day = _clamp_int(body.get("steps_per_day", 24), 4, 96, 24)
    economy_interval_steps = _clamp_int(
      body.get("economy_interval_steps", 24 if population_profile == "stanford_n3" else 1),
      1,
      240,
      24 if population_profile == "stanford_n3" else 1,
    )

    export_stanford_storage = _to_bool(body.get("export_stanford_storage", population_profile == "stanford_n3"))
    stanford_sim_code = str(body.get("stanford_sim_code", "")).strip()
    stanford_fork_sim_code = str(
      body.get("stanford_fork_sim_code", "base_the_ville_isabella_maria_klaus")
    ).strip() or "base_the_ville_isabella_maria_klaus"
    update_temp_files_for_simulator_home = _to_bool(
      body.get("update_temp_files_for_simulator_home", True)
    )
    enable_social_interaction = False
    enable_mobility = False
    enable_llm_agents = True
    llm_provider = str(body.get("llm_provider", "openai")).strip().lower()
    if llm_provider not in ["gemini", "openai"]:
      llm_provider = "openai"
    llm_model = str(body.get("llm_model", "")).strip()
    if not llm_model:
      llm_model = "gpt-4o-mini" if llm_provider == "openai" else "gemini-1.5-flash"
    llm_max_agents_per_step = _clamp_int(body.get("llm_max_agents_per_step", 3), 1, 300, 3)
    llm_temperature = _clamp_float(body.get("llm_temperature", 0.4), 0.0, 1.5, 0.4)
    llm_strict_mode = True
    llm_disable_rule_social = True
    enable_llm_economist = _to_bool(body.get("enable_llm_economist", True))
    llm_economist_every_steps = _clamp_int(body.get("llm_economist_every_steps", 24), 1, 240, 24)

    ts = datetime.datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    scenario_path = REPO_ROOT / FINANCIAL_TOWN_SCENARIOS[scenario_name]
    output_dir = REPO_ROOT / "financial_town" / "output" / "ui_runs" / f"{population_profile}_{scenario_name}_{ts}_{seed}"
    config = SimulationConfig(
      seed=seed,
      households=households,
      baseline_policy_path=str(REPO_ROOT / "financial_town" / "data" / "policies" / "base_policy.json"),
      scenario_policy_path=str(scenario_path),
      output_dir=str(output_dir),
      export_every_step=True,
      keep_step_snapshots=keep_snapshots,
      population_profile=population_profile,
      steps_per_day=steps_per_day,
      economy_interval_steps=economy_interval_steps,
      scripted_population_size=scripted_population_size,
      active_agent_name=active_agent_name,
      enable_social_interaction=enable_social_interaction,
      enable_mobility=enable_mobility,
      enable_llm_agents=enable_llm_agents,
      llm_provider=llm_provider,
      llm_model=llm_model,
      llm_temperature=llm_temperature,
      llm_max_agents_per_step=llm_max_agents_per_step,
      llm_strict_mode=llm_strict_mode,
      llm_disable_rule_social=llm_disable_rule_social,
      enable_llm_economist=enable_llm_economist,
      llm_economist_provider=llm_provider,
      llm_economist_model=llm_model,
      llm_economist_every_steps=llm_economist_every_steps,
      export_stanford_storage=export_stanford_storage,
      stanford_sim_code=stanford_sim_code,
      stanford_fork_sim_code=stanford_fork_sim_code,
      update_temp_files_for_simulator_home=update_temp_files_for_simulator_home,
    )
    simulator = FinancialTownSimulator(config)
    final_state = simulator.run(steps)
    final_sim_code = simulator.exporter.get_stanford_sim_code()

    metrics_history = final_state.metrics_history
    final_metrics = metrics_history[-1] if metrics_history else {}

    history_rows = []
    for m in metrics_history:
      history_rows.append({
        "step": m.get("step"),
        "employed_rate": m.get("employed_rate"),
        "total_debt": m.get("total_debt"),
        "loan_delinquency_rate": m.get("loan_delinquency_rate"),
        "fraudulent_txn_step": m.get("fraudulent_txn_step"),
        "interaction_events_step": m.get("interaction_events_step"),
        "llm_calls_step": m.get("llm_calls_step"),
        "llm_errors_step": m.get("llm_errors_step"),
        "inflation": m.get("macro_inflation"),
      })

    stanford_storage_dir = ""
    if final_sim_code:
      stanford_storage_dir = str(REPO_ROOT / "environment" / "frontend_server" / "storage" / final_sim_code)

    payload = {
      "ok": True,
      "run_context": {
        "scenario": scenario_name,
        "steps": steps,
        "households": len(final_state.households),
        "seed": seed,
        "population_profile": population_profile,
        "scripted_population_size": scripted_population_size,
        "active_agent_name": active_agent_name,
        "steps_per_day": steps_per_day,
        "economy_interval_steps": economy_interval_steps,
        "enable_llm_agents": enable_llm_agents,
        "llm_provider": llm_provider,
        "llm_model": llm_model,
        "enable_llm_economist": enable_llm_economist,
        "llm_economist_every_steps": llm_economist_every_steps,
        "output_dir": str(output_dir),
      },
      "final_metrics": final_metrics,
      "policy": {
        "income_tax_rate": final_state.policy.income_tax_rate,
        "payroll_tax_rate": final_state.policy.payroll_tax_rate,
        "central_bank_rate": final_state.policy.central_bank_rate,
        "layoff_rate": final_state.policy.layoff_rate,
        "housing_subsidy": final_state.policy.housing_subsidy,
        "fraud_detection_strength": final_state.policy.fraud_detection_strength,
      },
      "macro": {
        "gdp_growth": final_state.macro.gdp_growth,
        "inflation": final_state.macro.inflation,
        "unemployment_rate": final_state.macro.unemployment_rate,
        "housing_growth": final_state.macro.housing_growth,
      },
      "prices": {
        "consumer_price_index": float(getattr(final_state, "consumer_price_index", 1.0) or 1.0),
        "category_price_multipliers": getattr(final_state, "category_price_multipliers", {}) or {},
      },
      "history": history_rows,
      "top_risk_households": _household_risk_rows(final_state),
      "fraud_samples": _fraud_rows(final_state),
      "interaction_samples": _interaction_rows(final_state),
      "llm": {
        "enabled": enable_llm_agents,
        "provider": llm_provider,
        "model": llm_model,
        "stats": final_state.llm_stats,
      },
      "macro_events": final_state.macro_events[-20:],
      "stanford_export": {
        "enabled": export_stanford_storage,
        "sim_code": final_sim_code,
        "fork_sim_code": stanford_fork_sim_code,
        "storage_dir": stanford_storage_dir,
        "simulator_home_url": "/simulator_home" if export_stanford_storage else "",
        "replay_url": f"/replay/{final_sim_code}/0/" if final_sim_code else "",
      },
    }
    return JsonResponse(payload)

  except Exception as e:
    traceback.print_exc()
    return JsonResponse({
      "ok": False,
      "error": f"{type(e).__name__}: {str(e)}",
    }, status=500)


def demo(request, sim_code, step, play_speed="2"): 
  move_file = f"compressed_storage/{sim_code}/master_movement.json"
  meta_file = f"compressed_storage/{sim_code}/meta.json"
  step = int(step)
  play_speed_opt = {"1": 1, "2": 2, "3": 4,
                    "4": 8, "5": 16, "6": 32}
  if play_speed not in play_speed_opt: play_speed = 2
  else: play_speed = play_speed_opt[play_speed]

  # Loading the basic meta information about the simulation.
  meta = dict() 
  with open (meta_file) as json_file: 
    meta = json.load(json_file)

  sec_per_step = meta["sec_per_step"]
  start_datetime = datetime.datetime.strptime(meta["start_date"] + " 00:00:00", 
                                              '%B %d, %Y %H:%M:%S')
  for i in range(step): 
    start_datetime += datetime.timedelta(seconds=sec_per_step)
  start_datetime = start_datetime.strftime("%Y-%m-%dT%H:%M:%S")

  # Loading the movement file
  raw_all_movement = dict()
  with open(move_file) as json_file: 
    raw_all_movement = json.load(json_file)
 
  # Loading all names of the personas
  persona_names = dict()
  persona_names = []
  persona_names_set = set()
  for p in list(raw_all_movement["0"].keys()): 
    persona_names += [{"original": p, 
                       "underscore": p.replace(" ", "_"), 
                       "initial": p[0] + p.split(" ")[-1][0]}]
    persona_names_set.add(p)

  # <all_movement> is the main movement variable that we are passing to the 
  # frontend. Whereas we use ajax scheme to communicate steps to the frontend
  # during the simulation stage, for this demo, we send all movement 
  # information in one step. 
  all_movement = dict()

  # Preparing the initial step. 
  # <init_prep> sets the locations and descriptions of all agents at the
  # beginning of the demo determined by <step>. 
  init_prep = dict() 
  for int_key in range(step+1): 
    key = str(int_key)
    val = raw_all_movement[key]
    for p in persona_names_set: 
      if p in val: 
        init_prep[p] = val[p]
  persona_init_pos = dict()
  for p in persona_names_set: 
    persona_init_pos[p.replace(" ","_")] = init_prep[p]["movement"]
  all_movement[step] = init_prep

  # Finish loading <all_movement>
  for int_key in range(step+1, len(raw_all_movement.keys())): 
    all_movement[int_key] = raw_all_movement[str(int_key)]

  context = {"sim_code": sim_code,
             "step": step,
             "persona_names": persona_names,
             "persona_init_pos": json.dumps(persona_init_pos), 
             "all_movement": json.dumps(all_movement), 
             "start_datetime": start_datetime,
             "sec_per_step": sec_per_step,
             "play_speed": play_speed,
             "mode": "demo"}
  template = "demo/demo.html"

  return render(request, template, context)


def UIST_Demo(request): 
  return demo(request, "March20_the_ville_n25_UIST_RUN-step-1-141", 2160, play_speed="3")


def home(request):
  f_curr_sim_code = "temp_storage/curr_sim_code.json"
  f_curr_step = "temp_storage/curr_step.json"

  if not check_if_file_exists(f_curr_step): 
    context = {}
    template = "home/error_start_backend.html"
    return render(request, template, context)

  with open(f_curr_sim_code) as json_file:  
    sim_code = json.load(json_file)["sim_code"]
  
  with open(f_curr_step) as json_file:  
    step = json.load(json_file)["step"]

  os.remove(f_curr_step)

  persona_names = []
  persona_names_set = set()
  for i in find_filenames(f"storage/{sim_code}/personas", ""): 
    x = i.split("/")[-1].strip()
    if x[0] != ".": 
      persona_names += [[x, x.replace(" ", "_")]]
      persona_names_set.add(x)

  persona_init_pos = []
  curr_json = f"storage/{sim_code}/environment/{step}.json"
  if not os.path.exists(curr_json):
    file_count = []
    for i in find_filenames(f"storage/{sim_code}/environment", ".json"):
      x = i.split("/")[-1].strip()
      if x[0] != ".": 
        file_count += [int(x.split(".")[0])]
    curr_json = f'storage/{sim_code}/environment/{str(max(file_count))}.json'
  with open(curr_json) as json_file:  
    persona_init_pos_dict = json.load(json_file)
    for key, val in persona_init_pos_dict.items(): 
      if key in persona_names_set: 
        persona_init_pos += [[key, val["x"], val["y"]]]

  context = {"sim_code": sim_code,
             "step": step, 
             "persona_names": persona_names,
             "persona_init_pos": persona_init_pos,
             "mode": "simulate"}
  template = "home/home.html"
  return render(request, template, context)


def replay(request, sim_code, step): 
  sim_code = sim_code
  step = int(step)

  persona_names = []
  persona_names_set = set()
  for i in find_filenames(f"storage/{sim_code}/personas", ""): 
    x = i.split("/")[-1].strip()
    if x[0] != ".": 
      persona_names += [[x, x.replace(" ", "_")]]
      persona_names_set.add(x)

  persona_init_pos = []
  curr_json = f"storage/{sim_code}/environment/{step}.json"
  if not os.path.exists(curr_json):
    file_count = []
    for i in find_filenames(f"storage/{sim_code}/environment", ".json"):
      x = i.split("/")[-1].strip()
      if x[0] != ".": 
        file_count += [int(x.split(".")[0])]
    curr_json = f'storage/{sim_code}/environment/{str(max(file_count))}.json'
  with open(curr_json) as json_file:  
    persona_init_pos_dict = json.load(json_file)
    for key, val in persona_init_pos_dict.items(): 
      if key in persona_names_set: 
        persona_init_pos += [[key, val["x"], val["y"]]]

  context = {"sim_code": sim_code,
             "step": step,
             "persona_names": persona_names,
             "persona_init_pos": persona_init_pos, 
             "mode": "replay"}
  template = "home/home.html"
  return render(request, template, context)


def replay_persona_state(request, sim_code, step, persona_name): 
  sim_code = sim_code
  step = int(step)

  persona_name_underscore = persona_name
  persona_name = " ".join(persona_name.split("_"))
  memory = f"storage/{sim_code}/personas/{persona_name}/bootstrap_memory"
  if not os.path.exists(memory): 
    memory = f"compressed_storage/{sim_code}/personas/{persona_name}/bootstrap_memory"

  with open(memory + "/scratch.json") as json_file:  
    scratch = json.load(json_file)

  with open(memory + "/spatial_memory.json") as json_file:  
    spatial = json.load(json_file)

  with open(memory + "/associative_memory/nodes.json") as json_file:  
    associative = json.load(json_file)

  a_mem_event = []
  a_mem_chat = []
  a_mem_thought = []

  for count in range(len(associative.keys()), 0, -1): 
    node_id = f"node_{str(count)}"
    node_details = associative[node_id]

    if node_details["type"] == "event":
      a_mem_event += [node_details]

    elif node_details["type"] == "chat":
      a_mem_chat += [node_details]

    elif node_details["type"] == "thought":
      a_mem_thought += [node_details]
  
  context = {"sim_code": sim_code,
             "step": step,
             "persona_name": persona_name, 
             "persona_name_underscore": persona_name_underscore, 
             "scratch": scratch,
             "spatial": spatial,
             "a_mem_event": a_mem_event,
             "a_mem_chat": a_mem_chat,
             "a_mem_thought": a_mem_thought}
  template = "persona_state/persona_state.html"
  return render(request, template, context)


def path_tester(request):
  context = {}
  template = "path_tester/path_tester.html"
  return render(request, template, context)


def process_environment(request): 
  """
  <FRONTEND to BACKEND> 
  This sends the frontend visual world information to the backend server. 
  It does this by writing the current environment representation to 
  "storage/environment.json" file. 

  ARGS:
    request: Django request
  RETURNS: 
    HttpResponse: string confirmation message. 
  """
  # f_curr_sim_code = "temp_storage/curr_sim_code.json"
  # with open(f_curr_sim_code) as json_file:  
  #   sim_code = json.load(json_file)["sim_code"]

  data = json.loads(request.body)
  step = data["step"]
  sim_code = data["sim_code"]
  environment = data["environment"]

  with open(f"storage/{sim_code}/environment/{step}.json", "w") as outfile:
    outfile.write(json.dumps(environment, indent=2))

  return HttpResponse("received")


def update_environment(request): 
  """
  <BACKEND to FRONTEND> 
  This sends the backend computation of the persona behavior to the frontend
  visual server. 
  It does this by reading the new movement information from 
  "storage/movement.json" file.

  ARGS:
    request: Django request
  RETURNS: 
    HttpResponse
  """
  # f_curr_sim_code = "temp_storage/curr_sim_code.json"
  # with open(f_curr_sim_code) as json_file:  
  #   sim_code = json.load(json_file)["sim_code"]

  data = json.loads(request.body)
  step = data["step"]
  sim_code = data["sim_code"]

  response_data = {"<step>": -1}
  if (check_if_file_exists(f"storage/{sim_code}/movement/{step}.json")):
    with open(f"storage/{sim_code}/movement/{step}.json") as json_file: 
      response_data = json.load(json_file)
      response_data["<step>"] = step

  return JsonResponse(response_data)


def path_tester_update(request): 
  """
  Processing the path and saving it to path_tester_env.json temp storage for 
  conducting the path tester. 

  ARGS:
    request: Django request
  RETURNS: 
    HttpResponse: string confirmation message. 
  """
  data = json.loads(request.body)
  camera = data["camera"]

  with open(f"temp_storage/path_tester_env.json", "w") as outfile:
    outfile.write(json.dumps(camera, indent=2))

  return HttpResponse("received")

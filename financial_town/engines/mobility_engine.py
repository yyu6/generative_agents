"""Daily mobility routine: home -> work -> spending spots -> home."""

from __future__ import annotations

import random

from financial_town.models import EmploymentStatus
from financial_town.simulation.state import TownState


class MobilityEngine:
  def __init__(self, rng: random.Random, steps_per_day: int = 24):
    self.rng = rng
    self.steps_per_day = max(4, steps_per_day)

  def _move_one_step(self, curr_x: int, curr_y: int, target_x: int, target_y: int):
    x = curr_x
    y = curr_y
    if x < target_x:
      x += 1
    elif x > target_x:
      x -= 1
    elif y < target_y:
      y += 1
    elif y > target_y:
      y -= 1
    return x, y

  def _phase(self, step: int):
    hour = step % self.steps_per_day
    # Hourly schedule with commute windows.
    if 0 <= hour <= 6:
      return "home", "sleeping", "😴"
    if hour == 7:
      return "work", "commuting to work", "🚶"
    if 8 <= hour <= 16:
      return "work", "working", "💼"
    if hour == 17:
      return "spending", "commuting to evening spot", "🚶"
    if 18 <= hour <= 20:
      return "spending", "spending time in town", "🛍️"
    if hour == 21:
      return "home", "commuting home", "🚶"
    return "home", "resting at home", "🏠"

  def run_step(self, state: TownState) -> None:
    state.step_movements = {}
    state.step_chat = {}
    target_place, action, emoji = self._phase(state.step)

    for person in state.people.values():
      # Unemployed agents do not commute to workplace during daytime blocks.
      person_target = target_place
      if target_place == "work" and person.employment_status == EmploymentStatus.UNEMPLOYED:
        person_target = "spending" if (state.step % self.steps_per_day) >= 10 else "home"

      if person_target == "home":
        tx, ty = person.home_x, person.home_y
        address = person.home_address
      elif person_target == "work":
        tx, ty = person.work_x, person.work_y
        address = person.work_address
      else:
        tx, ty = person.spending_x, person.spending_y
        address = person.spending_address

      next_x, next_y = self._move_one_step(person.current_x, person.current_y, tx, ty)
      person.current_x, person.current_y = next_x, next_y

      if next_x == tx and next_y == ty:
        person.current_place = person_target
      else:
        person.current_place = f"commuting_to_{person_target}"

      state.step_movements[person.person_id] = {
        "movement": [next_x, next_y],
        "pronunciatio": emoji,
        "description": f"{action} @ {address}",
        "chat": None,
      }
      state.step_chat[person.person_id] = None

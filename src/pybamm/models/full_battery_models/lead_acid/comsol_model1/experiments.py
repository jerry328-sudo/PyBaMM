from __future__ import annotations

from dataclasses import dataclass

import pybamm


@dataclass(frozen=True)
class CycleControl:
    """Exact controller settings exported from the COMSOL Model1 benchmark."""

    charge_current_a: float = 0.32
    discharge_current_a: float = -0.8
    voltage_max_v: float = 2.4
    voltage_min_v: float = 1.75
    charge_current_cutoff_a: float = 0.01
    discharge_current_cutoff_a: float = -0.5
    rest_after_charge_s: float = 600.0
    rest_after_discharge_s: float = 600.0
    cycles_to_run: int = 2


DEFAULT_CYCLE_CONTROL = CycleControl()


def build_comsol_model1_controller_experiment(
    *,
    cycles_to_run: int | None = None,
    cycle: CycleControl = DEFAULT_CYCLE_CONTROL,
) -> pybamm.Experiment:
    """Build the full two-cycle CC-CV-rest-discharge-rest COMSOL controller."""

    repeat_count = cycle.cycles_to_run if cycles_to_run is None else cycles_to_run
    cycle_steps = (
        f"Charge at {cycle.charge_current_a} A until {cycle.voltage_max_v} V",
        f"Hold at {cycle.voltage_max_v} V until {cycle.charge_current_cutoff_a} A",
        f"Rest for {cycle.rest_after_charge_s} seconds",
        f"Discharge at {abs(cycle.discharge_current_a)} A until {cycle.voltage_min_v} V",
        f"Rest for {cycle.rest_after_discharge_s} seconds",
    )
    return pybamm.Experiment([cycle_steps] * repeat_count, period="60 seconds")


def build_comsol_model1_initialization_experiment(
    *,
    cycle: CycleControl = DEFAULT_CYCLE_CONTROL,
) -> pybamm.Experiment:
    """
    Build the one-second preload step used to initialise the DAE current fields.
    """

    return pybamm.Experiment(
        [f"Charge at {cycle.charge_current_a} A for 1 second"],
        period="1 second",
    )

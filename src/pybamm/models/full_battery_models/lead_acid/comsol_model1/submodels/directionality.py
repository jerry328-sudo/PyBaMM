from __future__ import annotations

import numpy as np

import pybamm


def stability_floor() -> pybamm.Scalar:
    """Small positive floor mirroring COMSOL's repeated `max(eps^2, ...)` guards."""

    return pybamm.Scalar(np.finfo(float).eps**2)


def discharge_branch_selector(current_a: pybamm.Symbol) -> pybamm.Symbol:
    """
    Map PyBaMM's current sign convention onto the COMSOL charge/discharge switch.

    In this repository's lead-acid workflow, discharge is `I > 0` and charge/rest is
    `I <= 0`.
    """

    return pybamm.maximum(pybamm.sign(current_a), 0)


def directional_parameter(
    current_a: pybamm.Symbol,
    *,
    charge_name: str,
    discharge_name: str,
) -> pybamm.Symbol:
    """Select a scalar parameter from charge/discharge variants using current sign."""

    discharge_weight = discharge_branch_selector(current_a)
    charge_weight = 1 - discharge_weight
    return (
        charge_weight * pybamm.Parameter(charge_name)
        + discharge_weight * pybamm.Parameter(discharge_name)
    )


def directional_function_parameter(
    current_a: pybamm.Symbol,
    *,
    charge_name: str,
    discharge_name: str,
    inputs: dict[str, pybamm.Symbol],
) -> pybamm.Symbol:
    """Select a function parameter from charge/discharge variants using current sign."""

    discharge_weight = discharge_branch_selector(current_a)
    charge_weight = 1 - discharge_weight
    return (
        charge_weight * pybamm.FunctionParameter(charge_name, inputs)
        + discharge_weight * pybamm.FunctionParameter(discharge_name, inputs)
    )


def normalized_porosity(
    domain: str,
    porosity: pybamm.Symbol,
) -> tuple[pybamm.Symbol, pybamm.Symbol]:
    """Return the COMSOL normalised porosity and remaining-capacity fraction."""

    floor = stability_floor()
    epsilon_max = pybamm.Parameter(f"Maximum porosity of {domain} electrode")
    epsilon_min = pybamm.Parameter(f"Minimum porosity of {domain} electrode")
    normalized = pybamm.maximum(
        floor,
        (porosity - epsilon_min) / (epsilon_max - epsilon_min),
    )
    remaining = pybamm.maximum(
        floor,
        -(porosity - epsilon_max) / (epsilon_max - epsilon_min),
    )
    return normalized, remaining


def main_reaction_surface_area(
    domain: str,
    current_a: pybamm.Symbol,
    porosity: pybamm.Symbol,
) -> pybamm.Symbol:
    """COMSOL main-reaction active area using the original single-parameter inputs."""

    Domain = domain.capitalize()
    floor = stability_floor()
    normalized, remaining = normalized_porosity(domain, porosity)

    charge_area = pybamm.maximum(
        floor,
        pybamm.Parameter(f"{Domain} electrode surface area to volume ratio [m-1]")
        * normalized ** pybamm.Parameter(f"{Domain} electrode charge morphology exponent")
        * remaining,
    )
    discharge_area = pybamm.maximum(
        floor,
        pybamm.Parameter(f"{Domain} electrode surface area to volume ratio [m-1]")
        * normalized
        ** pybamm.Parameter(f"{Domain} electrode morphology exponent"),
    )

    discharge_weight = discharge_branch_selector(current_a)
    return discharge_weight * discharge_area + (1 - discharge_weight) * charge_area


def side_reaction_surface_area(
    domain: str,
    current_a: pybamm.Symbol,
    porosity: pybamm.Symbol,
) -> pybamm.Symbol:
    """COMSOL side-reaction active area using the original single area parameter."""

    Domain = domain.capitalize()
    normalized, _ = normalized_porosity(domain, porosity)
    del current_a
    a = pybamm.Parameter(f"{Domain} electrode surface area to volume ratio [m-1]")
    return pybamm.maximum(stability_floor(), a * normalized)

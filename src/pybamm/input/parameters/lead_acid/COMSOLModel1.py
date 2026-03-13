from __future__ import annotations

from pathlib import Path

import numpy as np

import pybamm
from pybamm.parameters.process_parameter_data import process_1D_data

from .Sulzer2019 import get_parameter_values as get_sulzer2019_parameter_values


_DATA_DIR = Path(__file__).resolve().parent / "data" / "comsol_model1"


def _load_material_table(filename: str) -> np.ndarray:
    """Load one COMSOL material table via PyBaMM's standard 1D data loader."""

    _, data = process_1D_data(filename, path=_DATA_DIR)
    x_data = np.asarray(data[0][0], dtype=float)
    y_data = np.asarray(data[1], dtype=float)
    return np.column_stack((x_data, y_data))


def _linear_extrapolation(x_value: float, table: np.ndarray) -> float:
    """Evaluate a 2-column table with linear interpolation and linear extrapolation."""

    x_data = table[:, 0]
    y_data = table[:, 1]

    if x_value <= x_data[0]:
        x_left, y_left = x_data[0], y_data[0]
        x_right, y_right = x_data[1], y_data[1]
    elif x_value >= x_data[-1]:
        x_left, y_left = x_data[-2], y_data[-2]
        x_right, y_right = x_data[-1], y_data[-1]
    else:
        return float(np.interp(x_value, x_data, y_data))

    slope = (y_right - y_left) / (x_right - x_left)
    return float(y_left + slope * (x_value - x_left))


def _clip_to_table_bounds(
    child: pybamm.Symbol | float,
    table: np.ndarray,
    *,
    clip_lower_bound: bool = False,
    clip_upper_bound: bool = False,
) -> pybamm.Symbol | float:
    """Clamp interpolant inputs to the tabulated range before evaluation."""

    lower_bound = float(table[0, 0])
    upper_bound = float(table[-1, 0])

    if isinstance(child, (int, float, np.floating)):
        value = float(child)
        if clip_lower_bound:
            value = max(value, lower_bound)
        if clip_upper_bound:
            value = min(value, upper_bound)
        return value

    clipped = child
    if clip_lower_bound:
        clipped = pybamm.maximum(clipped, lower_bound)
    if clip_upper_bound:
        clipped = pybamm.minimum(clipped, upper_bound)
    return clipped


def _table_interpolant(
    table: np.ndarray,
    child: pybamm.Symbol | float,
    *,
    clip_child_to_lower_bound: bool = False,
    clip_child_to_upper_bound: bool = False,
    name: str,
) -> pybamm.Interpolant | float:
    """
    Build a PyBaMM interpolant for one COMSOL material table.

    Symbolic inputs keep the interpolant inside the expression tree while numeric
    inputs are evaluated directly so tests can assert exact values.
    """

    interpolant_child = (
        _clip_to_table_bounds(
            child,
            table,
            clip_lower_bound=clip_child_to_lower_bound,
            clip_upper_bound=clip_child_to_upper_bound,
        )
        if (clip_child_to_lower_bound or clip_child_to_upper_bound)
        else child
    )

    if isinstance(interpolant_child, (int, float, np.floating)):
        return _linear_extrapolation(float(interpolant_child), table)

    return pybamm.Interpolant(
        table[:, 0],
        table[:, 1],
        [interpolant_child],
        name=name,
        interpolator="linear",
        extrapolate=True,
    )


def electrolyte_concentration_from_molality(
    molality: pybamm.Symbol | float,
    *,
    molar_volume_electrolyte: float,
    molar_volume_water: float,
    molar_mass_water: float,
) -> pybamm.Symbol:
    """
    Convert PyBaMM's lead-acid molality back to the concentration used by COMSOL.
    """

    return (
        molality * molar_mass_water
        / (molar_volume_water + molality * molar_mass_water * molar_volume_electrolyte)
    )


DIFFUSIVITY_TABLE = _load_material_table("sulfuric_acid_diffusivity_Dl_int1(c).csv")
CONDUCTIVITY_TABLE = _load_material_table("sulfuric_acid_conductivity_sigmal_int1(c).csv")
NEGATIVE_OCP_TABLE = _load_material_table("pb_equilibrium_potential_Eeq_Pb_int1(c).csv")
POSITIVE_OCP_TABLE = _load_material_table("pbo2_equilibrium_potential_Eeq_PbO2_int1(c).csv")


def electrolyte_diffusivity(c_e: pybamm.Symbol | float) -> pybamm.Symbol | float:
    """Electrolyte diffusivity from the COMSOL materials database."""

    return _table_interpolant(
        DIFFUSIVITY_TABLE,
        c_e,
        clip_child_to_lower_bound=True,
        clip_child_to_upper_bound=True,
        name="comsol_model1_diffusivity",
    )


def electrolyte_conductivity(c_e: pybamm.Symbol | float) -> pybamm.Symbol | float:
    """Electrolyte conductivity from the COMSOL materials database."""

    return _table_interpolant(
        CONDUCTIVITY_TABLE,
        c_e,
        clip_child_to_lower_bound=True,
        clip_child_to_upper_bound=True,
        name="comsol_model1_conductivity",
    )


def cation_transference_number(c_e: pybamm.Symbol | float) -> float:
    """Return COMSOL Model1's constant cation transference number."""

    del c_e
    return 0.72


def darken_thermodynamic_factor(c_e: pybamm.Symbol | float) -> float:
    """Return the constant thermodynamic factor used by the COMSOL case."""

    del c_e
    return 1.0


def negative_open_circuit_potential(
    molality: pybamm.Symbol | float,
) -> pybamm.Symbol | float:
    """Negative-electrode OCP table evaluated against COMSOL concentration."""

    concentration = electrolyte_concentration_from_molality(
        molality,
        molar_volume_electrolyte=4.5e-5,
        molar_volume_water=1.75e-5,
        molar_mass_water=0.01801,
    )
    return _table_interpolant(
        NEGATIVE_OCP_TABLE,
        concentration,
        clip_child_to_lower_bound=True,
        clip_child_to_upper_bound=True,
        name="comsol_model1_negative_ocp",
    )


def positive_open_circuit_potential(
    molality: pybamm.Symbol | float,
) -> pybamm.Symbol | float:
    """Positive-electrode OCP table evaluated against COMSOL concentration."""

    concentration = electrolyte_concentration_from_molality(
        molality,
        molar_volume_electrolyte=4.5e-5,
        molar_volume_water=1.75e-5,
        molar_mass_water=0.01801,
    )
    return _table_interpolant(
        POSITIVE_OCP_TABLE,
        concentration,
        clip_child_to_lower_bound=True,
        clip_child_to_upper_bound=True,
        name="comsol_model1_positive_ocp",
    )


def negative_exchange_current_density(
    c_e: pybamm.Symbol | float,
    temperature_k: pybamm.Symbol | float,
) -> pybamm.Symbol:
    """Main negative-reaction exchange current density from COMSOL Model1."""

    del temperature_k
    return 0.03 * (c_e / 4890.0) ** 0.0


def positive_exchange_current_density(
    c_e: pybamm.Symbol | float,
    temperature_k: pybamm.Symbol | float,
) -> pybamm.Symbol:
    """Main positive-reaction exchange current density from COMSOL Model1."""

    del temperature_k
    return 0.003 * (c_e / 4890.0) ** 0.0


def positive_oxygen_exchange_current_density(
    c_e: pybamm.Symbol | float,
    temperature_k: pybamm.Symbol | float,
) -> pybamm.Symbol:
    """Positive OER exchange current density from COMSOL Model1."""

    del temperature_k
    return 1.0e-23 * (c_e / 4890.0) ** 2


def negative_hydrogen_exchange_current_density(
    c_e: pybamm.Symbol | float,
    temperature_k: pybamm.Symbol | float,
) -> pybamm.Symbol:
    """Negative HER exchange current density from COMSOL Model1."""

    del temperature_k
    return 1.0e-9 * c_e / 4890.0


def get_parameter_values() -> dict[str, object]:
    """
    Return the COMSOL-aligned lead-acid parameter set for the Model1 benchmark.

    The parameterisation starts from PyBaMM's standard Sulzer2019 set and then
    overrides only the values needed to reproduce the COMSOL-exported geometry,
    transport tables and reaction kinetics.
    """

    values = get_sulzer2019_parameter_values().copy()
    values.update(
        {
            "Negative current collector thickness [m]": 0.0,
            "Positive current collector thickness [m]": 0.0,
            "Ambient temperature [K]": 298.0,
            "Initial temperature [K]": 298.0,
            "Reference temperature [K]": 298.0,
            "Number of electrodes connected in parallel to make a cell": 1.0,
            "Number of cells connected in series to make a battery": 1.0,
            "Lower voltage cut-off [V]": 1.75,
            "Upper voltage cut-off [V]": 2.4,
            "Initial concentration in electrolyte [mol.m-3]": 4890.0,
            "Electrode height [m]": 0.07,
            "Electrode width [m]": 0.038,
            "Negative electrode thickness [m]": 0.0016,
            "Separator thickness [m]": 0.0026,
            "Positive electrode thickness [m]": 0.0020,
            "Negative electrode Bruggeman coefficient (electrolyte)": 1.5,
            "Positive electrode Bruggeman coefficient (electrolyte)": 1.5,
            "Separator Bruggeman coefficient (electrolyte)": 3.53,
            "Negative electrode Bruggeman coefficient (electrode)": 0.5,
            "Positive electrode Bruggeman coefficient (electrode)": 0.5,
            "Electrolyte conductivity [S.m-1]": electrolyte_conductivity,
            "Electrolyte diffusivity [m2.s-1]": electrolyte_diffusivity,
            "Cation transference number": cation_transference_number,
            "Thermodynamic factor": 1.0,
            "Darken thermodynamic factor": darken_thermodynamic_factor,
            "Maximum porosity of negative electrode": 0.53,
            "Maximum porosity of separator": 0.73,
            "Separator electrolyte source relaxation rate [s-1]": 0.0,
            "Maximum porosity of positive electrode": 0.53,
            "Initial porosity of negative electrode": 0.52,
            "Initial porosity of separator": 0.73,
            "Initial porosity of positive electrode": 0.52,
            "Minimum porosity of negative electrode": 0.3066,
            "Minimum porosity of positive electrode": 0.3466,
            "Negative electrode conductivity [S.m-1]": 4.8e6,
            "Positive electrode conductivity [S.m-1]": 8.0e3,
            "Negative electrode surface area to volume ratio [m-1]": 2.3e6,
            "Positive electrode surface area to volume ratio [m-1]": 2.3e7,
            "Negative electrode morphology exponent": 2.0,
            "Positive electrode morphology exponent": 2.0,
            "Negative electrode charge morphology exponent": 1.0,
            "Positive electrode charge morphology exponent": 1.0,
            "Negative electrode double-layer capacity [F.m-2]": 0.2,
            "Positive electrode double-layer capacity [F.m-2]": 0.2,
            "Negative electrode exchange-current density [A.m-2]": negative_exchange_current_density,
            "Positive electrode exchange-current density [A.m-2]": positive_exchange_current_density,
            "Positive electrode oxygen exchange-current density [A.m-2]": positive_oxygen_exchange_current_density,
            "Negative electrode hydrogen exchange-current density [A.m-2]": negative_hydrogen_exchange_current_density,
            "Negative electrode reference exchange-current density (hydrogen) [A.m-2]": 1.0e-9,
            "Negative electrode reference exchange-current density (oxygen) [A.m-2]": 0.0,
            "Positive electrode reference exchange-current density (hydrogen) [A.m-2]": 0.0,
            "Electrons in oxygen reaction": 2.0,
            "Electrons in hydrogen reaction": 2.0,
            "Signed stoichiometry of cations (oxygen reaction)": -2.0,
            "Signed stoichiometry of water (oxygen reaction)": 1.0,
            "Signed stoichiometry of oxygen (oxygen reaction)": 1.0,
            "Signed stoichiometry of cations (hydrogen reaction)": -2.0,
            "Negative electrode open-circuit potential [V]": negative_open_circuit_potential,
            "Positive electrode open-circuit potential [V]": positive_open_circuit_potential,
            "Oxygen reference OCP vs SHE [V]": 1.23,
            "Hydrogen reference OCP vs SHE [V]": 0.0,
            "Positive oxygen anodic transfer coefficient": 2.0,
            "Positive oxygen cathodic transfer coefficient": 2.0,
            "Negative hydrogen anodic transfer coefficient": 0.5,
            "Negative hydrogen cathodic transfer coefficient": 0.5,
            "Positive oxygen HSO4- stoichiometry": 0.0,
            "Negative hydrogen HSO4- stoichiometry": 0.0,
            "Negative electrode anodic transfer coefficient": 1.55,
            "Negative electrode cathodic transfer coefficient": 0.45,
            "Positive electrode anodic transfer coefficient": 1.21,
            "Positive electrode cathodic transfer coefficient": 0.79,
        }
    )
    return values

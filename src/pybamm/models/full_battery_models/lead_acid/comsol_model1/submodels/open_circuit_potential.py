from __future__ import annotations

import pybamm
from pybamm.models.submodels.interface.open_circuit_potential.base_ocp import BaseOpenCircuitPotential

from .directionality import directional_function_parameter, directional_parameter


class ComsolMainReactionOpenCircuitPotential(BaseOpenCircuitPotential):
    """Lead-acid main-reaction OCP with charge/discharge-specific COMSOL parameters."""

    def get_coupled_variables(self, variables):
        domain, Domain = self.domain_Domain
        c_e = variables[f"{Domain} electrolyte concentration [mol.m-3]"]
        current_a = variables["Current [A]"]

        if isinstance(c_e, pybamm.Broadcast):
            c_e = c_e.orphans[0]

        ocp_surf = directional_function_parameter(
            current_a,
            charge_name=f"Charge {Domain} electrode open-circuit potential [V]",
            discharge_name=f"Discharge {Domain} electrode open-circuit potential [V]",
            inputs={"Electrolyte molar mass [mol.kg-1]": self.param.m(c_e)},
        )

        c_e_av = variables["X-averaged electrolyte concentration [mol.m-3]"]
        ocp_bulk = directional_function_parameter(
            current_a,
            charge_name=f"Charge {Domain} electrode open-circuit potential [V]",
            discharge_name=f"Discharge {Domain} electrode open-circuit potential [V]",
            inputs={"Electrolyte molar mass [mol.kg-1]": self.param.m(c_e_av)},
        )

        dUdT = pybamm.Scalar(0)
        variables.update(self._get_standard_ocp_variables(ocp_surf, ocp_bulk, dUdT))
        return variables


class ComsolConstantSideReactionOpenCircuitPotential(BaseOpenCircuitPotential):
    """Constant side-reaction equilibrium potential for COMSOL's `per2` features."""

    def __init__(
        self,
        param,
        domain: str,
        *,
        reaction: str,
        reaction_name: str,
        potential_parameter_name: str,
        options,
        phase: str = "primary",
    ):
        super().__init__(param, domain, "lead-acid oxygen", options, phase)
        self.reaction = reaction
        self.reaction_name = reaction_name
        self.potential_parameter_name = potential_parameter_name

    def get_coupled_variables(self, variables):
        ocp_surf = directional_parameter(
            variables["Current [A]"],
            charge_name=f"Charge {self.potential_parameter_name}",
            discharge_name=f"Discharge {self.potential_parameter_name}",
        )
        ocp_bulk = ocp_surf
        dUdT = pybamm.Scalar(0)
        variables.update(self._get_standard_ocp_variables(ocp_surf, ocp_bulk, dUdT))
        return variables

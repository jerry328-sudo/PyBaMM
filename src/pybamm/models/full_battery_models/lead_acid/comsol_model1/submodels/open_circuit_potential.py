from __future__ import annotations

import pybamm
from pybamm.models.submodels.interface.open_circuit_potential.base_ocp import BaseOpenCircuitPotential


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
        ocp_surf = pybamm.Parameter(self.potential_parameter_name)
        ocp_bulk = ocp_surf
        dUdT = pybamm.Scalar(0)
        variables.update(self._get_standard_ocp_variables(ocp_surf, ocp_bulk, dUdT))
        return variables

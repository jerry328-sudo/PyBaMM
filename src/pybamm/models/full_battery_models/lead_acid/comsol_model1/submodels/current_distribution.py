from __future__ import annotations

import pybamm

from pybamm.models.submodels.electrolyte_diffusion.base_electrolyte_diffusion import (
    BaseElectrolyteDiffusion,
)
from pybamm.models.submodels.external_circuit.base_external_circuit import (
    BaseModel as BaseExternalCircuitModel,
)
from pybamm.models.submodels.porosity.base_porosity import BaseModel as BasePorosityModel

from .electrolyte import ComsolFormSurfaceFormDifferential
from .kinetics import ComsolMainReactionKinetics


class ComsolCurrentDistributionConstantPorosity(BasePorosityModel):
    """Broadcast the configured initial porosity fields as constant domain variables."""

    def get_fundamental_variables(self):
        eps_dict = {
            "negative electrode": pybamm.FullBroadcast(
                pybamm.Parameter("Initial porosity of negative electrode"),
                "negative electrode",
                "current collector",
            ),
            "separator": pybamm.FullBroadcast(
                pybamm.Parameter("Initial porosity of separator"),
                "separator",
                "current collector",
            ),
            "positive electrode": pybamm.FullBroadcast(
                pybamm.Parameter("Initial porosity of positive electrode"),
                "positive electrode",
                "current collector",
            ),
        }
        depsdt_dict = {
            domain: pybamm.FullBroadcast(0, domain, "current collector")
            for domain in eps_dict
        }
        variables = self._get_standard_porosity_variables(eps_dict)
        variables.update(self._get_standard_porosity_change_variables(depsdt_dict))
        return variables

    def add_events_from(self, variables):
        del variables
        return


class ComsolCurrentDistributionConstantConcentration(BaseElectrolyteDiffusion):
    """Broadcast the initial electrolyte concentration as a constant field."""

    def get_fundamental_variables(self):
        c_e_dict = {
            domain: pybamm.FullBroadcast(
                self.param.c_e_init,
                domain,
                "current collector",
            )
            for domain in self.options.whole_cell_domains
        }
        variables = self._get_standard_concentration_variables(c_e_dict)
        variables["Electrolyte concentration concatenation [mol.m-3]"] = variables[
            "Electrolyte concentration [mol.m-3]"
        ]
        return variables

    def get_coupled_variables(self, variables):
        eps_c_e_dict = {}
        for domain in self.options.whole_cell_domains:
            porosity_name = domain.capitalize()
            concentration_name = domain.split()[0].capitalize()
            eps_k = variables[f"{porosity_name} porosity"]
            c_e_k = variables[f"{concentration_name} electrolyte concentration [mol.m-3]"]
            eps_c_e_dict[domain] = eps_k * c_e_k

        zero_flux = pybamm.FullBroadcastToEdges(
            0,
            [domain for domain in self.options.whole_cell_domains],
            "current collector",
        )
        variables = variables.copy()
        variables.update(
            self._get_standard_porosity_times_concentration_variables(eps_c_e_dict)
        )
        variables.update(self._get_standard_flux_variables(zero_flux))
        variables.update(
            {
                "Electrolyte diffusion flux [mol.m-2.s-1]": zero_flux,
                "Electrolyte migration flux [mol.m-2.s-1]": zero_flux,
                "Electrolyte convection flux [mol.m-2.s-1]": zero_flux,
            }
        )
        return variables

    def set_boundary_conditions(self, variables):
        c_e = variables["Electrolyte concentration [mol.m-3]"]
        zero = pybamm.Scalar(0)
        self.boundary_conditions = {
            c_e: {"left": (zero, "Neumann"), "right": (zero, "Neumann")}
        }

    def add_events_from(self, variables):
        del variables
        return


class ComsolCurrentDistributionInitializationSurfaceForm(
    ComsolFormSurfaceFormDifferential
):
    """
    Stationary surface-form closure for COMSOL's current-distribution initializer.

    COMSOL's `CurrentDistributionInitialization` study does not advance the
    double-layer state. Instead, it enforces local equilibrium

        phis - phil = Eeq

    while solving the current distribution under the applied boundary current.
    This submodel keeps the COMSOL liquid-current law from the main model, but
    replaces the transient double-layer equation with a local algebraic
    equilibrium constraint.
    """

    def set_rhs(self, variables):
        del variables
        return

    def set_algebraic(self, variables):
        if self.domain == "separator":
            return

        Domain = self.domain.capitalize()
        delta_phi = variables[f"{Domain} electrode surface potential difference [V]"]
        ocp = variables[f"{Domain} electrode open-circuit potential [V]"]

        if isinstance(ocp, pybamm.Broadcast):
            ocp = ocp.orphans[0]

        self.algebraic[delta_phi] = delta_phi - ocp

    def set_boundary_conditions(self, variables):
        if self.domain == "separator":
            return

        Domain = self.domain.capitalize()
        delta_phi = variables[f"{Domain} electrode surface potential difference [V]"]
        zero = pybamm.Scalar(0)
        self.boundary_conditions = {
            delta_phi: {"left": (zero, "Neumann"), "right": (zero, "Neumann")}
        }


class ComsolNegativeHydrogenNoReaction(pybamm.kinetics.NoReaction):
    """
    Zero-current hydrogen branch used only during current-distribution initialisation.
    """

    def __init__(self, param, domain, options, phase="primary"):
        super().__init__(param, domain, "lead-acid oxygen", options, phase)
        self.reaction = "lead-acid hydrogen"
        self.reaction_name = "hydrogen "


class ComsolMainReactionInitialization(ComsolMainReactionKinetics):
    """
    Main-reaction interface for CDI: publish equilibrium overpotential fields with zero current.
    """

    def _get_kinetics(self, j0, ne, eta_r, T, u):
        del j0, ne, T, u
        return pybamm.zeros_like(eta_r)

    def _get_average_total_interfacial_current_density(self, variables):
        del variables
        zero = pybamm.Scalar(0)
        return zero, zero


class ComsolZeroThroughput(BaseExternalCircuitModel):
    """Expose zero-valued throughput variables so CDI stays purely algebraic."""

    def get_fundamental_variables(self):
        zero = pybamm.Scalar(0)
        return {
            "Discharge capacity [A.h]": zero,
            "Throughput capacity [A.h]": zero,
            "Discharge energy [W.h]": zero,
            "Throughput energy [W.h]": zero,
        }

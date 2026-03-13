from __future__ import annotations

import numpy as np

import pybamm
from pybamm.models.submodels.interface.kinetics.butler_volmer import AsymmetricButlerVolmer


def _stability_floor() -> pybamm.Scalar:
    """Small positive floor mirroring COMSOL's repeated `max(eps^2, ...)` guards."""

    return pybamm.Scalar(np.finfo(float).eps**2)


def _normalized_porosity(
    domain: str,
    porosity: pybamm.Symbol,
) -> tuple[pybamm.Symbol, pybamm.Symbol]:
    """
    Return the COMSOL normalised porosity and remaining-capacity fraction.
    """

    floor = _stability_floor()
    epsilon_max = pybamm.Parameter(f"Maximum porosity of {domain} electrode")
    epsilon_min = pybamm.Parameter(f"Minimum porosity of {domain} electrode")
    normalized_porosity = pybamm.maximum(
        floor,
        (porosity - epsilon_min) / (epsilon_max - epsilon_min),
    )
    remaining_capacity_fraction = pybamm.maximum(
        floor,
        -(porosity - epsilon_max) / (epsilon_max - epsilon_min),
    )
    return normalized_porosity, remaining_capacity_fraction


def _discharge_branch_selector(current_a: pybamm.Symbol) -> pybamm.Symbol:
    """
    Map PyBaMM current sign convention onto COMSOL's charge/discharge branch switch.
    """

    return pybamm.maximum(pybamm.sign(current_a), 0)


class _ComsolKineticsMixin:
    """Shared utilities for the COMSOL Model1 kinetics submodels."""

    def _explicit_butler_volmer(
        self,
        j0: pybamm.Symbol,
        eta_r: pybamm.Symbol,
        T: pybamm.Symbol,
        utilisation: pybamm.Symbol,
        *,
        anodic_parameter: str,
        cathodic_parameter: str,
    ) -> pybamm.Symbol:
        """
        Evaluate Butler-Volmer with independent anodic and cathodic coefficients.
        """

        alpha_a = pybamm.Parameter(anodic_parameter)
        alpha_c = pybamm.Parameter(cathodic_parameter)
        Feta_RT = self.param.F * eta_r / (self.param.R * T)
        return utilisation * j0 * (
            pybamm.exp(alpha_a * Feta_RT) - pybamm.exp(-alpha_c * Feta_RT)
        )

    def _publish_area_weighted_volumetric_current_density(
        self,
        variables: dict[str, pybamm.Symbol],
        area: pybamm.Symbol,
    ) -> dict[str, pybamm.Symbol]:
        """
        Publish volumetric interfacial current densities using COMSOL's area law.
        """

        domain, Domain = self.domain_Domain
        if self.options.electrode_types[domain] == "planar":
            return variables

        interfacial_current_density = variables[
            f"{Domain} electrode {self.reaction_name}interfacial current density [A.m-2]"
        ]
        volumetric_current_density = area * interfacial_current_density
        variables.update(
            {
                f"{Domain} electrode {self.reaction_name}volumetric interfacial current density [A.m-3]": volumetric_current_density,
                f"X-averaged {domain} electrode {self.reaction_name}volumetric interfacial current density [A.m-3]": pybamm.x_average(
                    volumetric_current_density
                ),
            }
        )
        return variables


class ComsolMainReactionKinetics(_ComsolKineticsMixin, AsymmetricButlerVolmer):
    """Main lead-acid kinetics with the COMSOL active-area branch logic."""

    def _surface_area(self, variables: dict[str, pybamm.Symbol]) -> pybamm.Symbol:
        domain, Domain = self.domain_Domain
        porosity = variables[f"{Domain} electrode porosity"]
        current_a = variables["Current [A]"]

        floor = _stability_floor()
        a_max = pybamm.Parameter(f"{Domain} electrode surface area to volume ratio [m-1]")
        discharge_morphology = pybamm.Parameter(f"{Domain} electrode morphology exponent")
        charge_morphology = pybamm.Parameter(f"{Domain} electrode charge morphology exponent")
        normalized_porosity, remaining_capacity_fraction = _normalized_porosity(domain, porosity)

        discharge_area = pybamm.maximum(
            floor,
            a_max * normalized_porosity**discharge_morphology,
        )
        charge_area = pybamm.maximum(
            floor,
            a_max * normalized_porosity**charge_morphology * remaining_capacity_fraction,
        )

        discharge_selector = _discharge_branch_selector(current_a)
        return discharge_selector * discharge_area + (1 - discharge_selector) * charge_area

    def _get_kinetics(self, j0, ne, eta_r, T, u):
        del ne
        Domain = self.domain_Domain[1]
        return self._explicit_butler_volmer(
            j0,
            eta_r,
            T,
            u,
            anodic_parameter=f"{Domain} electrode anodic transfer coefficient",
            cathodic_parameter=f"{Domain} electrode cathodic transfer coefficient",
        )

    def _get_average_total_interfacial_current_density(
        self,
        variables: dict[str, pybamm.Symbol],
    ) -> tuple[pybamm.Symbol, pybamm.Symbol]:
        """Average total current using the COMSOL main-reaction area field."""

        i_boundary_cc = variables["Current collector current density [A.m-2]"]
        if self.options.electrode_types[self.domain] == "planar":
            return i_boundary_cc, i_boundary_cc

        sgn = 1 if self.domain == "negative" else -1
        a_av = pybamm.x_average(self._surface_area(variables))
        a_j_total_average = sgn * i_boundary_cc / self.domain_param.L
        j_total_average = a_j_total_average / a_av
        return j_total_average, a_j_total_average

    def _get_standard_volumetric_current_density_variables(
        self,
        variables: dict[str, pybamm.Symbol],
    ) -> dict[str, pybamm.Symbol]:
        return self._publish_area_weighted_volumetric_current_density(
            variables,
            self._surface_area(variables),
        )


class ComsolPositiveOxygenKinetics(_ComsolKineticsMixin, AsymmetricButlerVolmer):
    """Positive oxygen evolution kinetics matching COMSOL's `per2` definition."""

    def _surface_area(self, variables: dict[str, pybamm.Symbol]) -> pybamm.Symbol:
        domain, Domain = self.domain_Domain
        porosity = variables[f"{Domain} electrode porosity"]
        normalized_porosity, _ = _normalized_porosity(domain, porosity)
        a_max = pybamm.Parameter(f"{Domain} electrode surface area to volume ratio [m-1]")
        return pybamm.maximum(_stability_floor(), a_max * normalized_porosity)

    def _get_kinetics(self, j0, ne, eta_r, T, u):
        del ne
        return self._explicit_butler_volmer(
            j0,
            eta_r,
            T,
            u,
            anodic_parameter="Positive oxygen anodic transfer coefficient",
            cathodic_parameter="Positive oxygen cathodic transfer coefficient",
        )

    def _get_standard_total_interfacial_current_variables(self, j_tot_av, a_j_tot_av):
        del j_tot_av, a_j_tot_av
        return {}

    def _get_standard_volumetric_current_density_variables(
        self,
        variables: dict[str, pybamm.Symbol],
    ) -> dict[str, pybamm.Symbol]:
        return self._publish_area_weighted_volumetric_current_density(
            variables,
            self._surface_area(variables),
        )


class ComsolNegativeHydrogenKinetics(_ComsolKineticsMixin, AsymmetricButlerVolmer):
    """Negative hydrogen evolution kinetics for the COMSOL side-reaction topology."""

    def __init__(self, param, domain, options, phase="primary"):
        super().__init__(param, domain, "lead-acid oxygen", options, phase)
        self.reaction = "lead-acid hydrogen"
        self.reaction_name = "hydrogen "

    def _surface_area(self, variables: dict[str, pybamm.Symbol]) -> pybamm.Symbol:
        domain, Domain = self.domain_Domain
        porosity = variables[f"{Domain} electrode porosity"]
        normalized_porosity, _ = _normalized_porosity(domain, porosity)
        a_max = pybamm.Parameter(f"{Domain} electrode surface area to volume ratio [m-1]")
        return pybamm.maximum(_stability_floor(), a_max * normalized_porosity)

    def _get_exchange_current_density(self, variables):
        Domain = self.domain_Domain[1]
        c_e = variables[f"{Domain} electrolyte concentration [mol.m-3]"]
        T = variables[f"{Domain} electrode temperature [K]"]
        if isinstance(c_e, pybamm.Broadcast) and isinstance(T, pybamm.Broadcast):
            c_e = c_e.orphans[0]
            T = T.orphans[0]
        return pybamm.FunctionParameter(
            "Negative electrode hydrogen exchange-current density [A.m-2]",
            {"Electrolyte concentration [mol.m-3]": c_e, "Temperature [K]": T},
        )

    def _get_number_of_electrons_in_reaction(self):
        return self.param.ne_Hy

    def _get_kinetics(self, j0, ne, eta_r, T, u):
        del ne
        return self._explicit_butler_volmer(
            j0,
            eta_r,
            T,
            u,
            anodic_parameter="Negative hydrogen anodic transfer coefficient",
            cathodic_parameter="Negative hydrogen cathodic transfer coefficient",
        )

    def _get_standard_total_interfacial_current_variables(self, j_tot_av, a_j_tot_av):
        del j_tot_av, a_j_tot_av
        return {}

    def _get_standard_volumetric_current_density_variables(
        self,
        variables: dict[str, pybamm.Symbol],
    ) -> dict[str, pybamm.Symbol]:
        return self._publish_area_weighted_volumetric_current_density(
            variables,
            self._surface_area(variables),
        )

from __future__ import annotations

import pybamm


def side_reaction_electrolyte_source_from_current_density(
    volumetric_current_density_a_m3,
    *,
    electrons_in_reaction,
    cation_stoichiometry,
):
    """
    Map side-reaction current to the electrolyte source accumulator used by PyBaMM.
    """

    return -volumetric_current_density_a_m3 * cation_stoichiometry / electrons_in_reaction


class ComsolLeadAcidTotalInterfacialCurrent(pybamm.interface.TotalInterfacialCurrent):
    """
    Total interfacial current for the COMSOL side-reaction topology.

    PyBaMM's stock lead-acid implementation couples positive oxygen evolution to
    negative oxygen reduction. COMSOL Model1 instead couples positive oxygen
    evolution to negative hydrogen evolution.
    """

    def __init__(self, param, options):
        super().__init__(param, "lead-acid", options)

    def _get_side_reaction_electrolyte_source_term(self, variables, domain, reaction_name, a_j_k):
        del variables, domain
        if reaction_name == "oxygen ":
            electrons = pybamm.Parameter("Electrons in oxygen reaction")
            cation_stoichiometry = pybamm.Parameter(
                "Signed stoichiometry of cations (oxygen reaction)"
            )
        else:
            electrons = pybamm.Parameter("Electrons in hydrogen reaction")
            cation_stoichiometry = pybamm.Parameter(
                "Signed stoichiometry of cations (hydrogen reaction)"
            )

        return side_reaction_electrolyte_source_from_current_density(
            a_j_k,
            electrons_in_reaction=electrons,
            cation_stoichiometry=cation_stoichiometry,
        )

    def _get_coupled_variables_by_phase_and_domain(self, variables, domain, phase_name):
        Domain = domain.capitalize()

        if domain == "negative":
            reaction_names = ["", "hydrogen "]
        elif domain == "positive":
            reaction_names = ["", "oxygen "]
        else:
            reaction_names = [""]

        new_variables = variables.copy()
        new_variables.update(
            {
                f"Sum of {domain} electrode {phase_name}electrolyte reaction source terms [A.m-3]": 0,
                f"Sum of x-averaged {domain} electrode {phase_name}electrolyte reaction source terms [A.m-3]": 0,
                f"Sum of {domain} electrode {phase_name}volumetric interfacial current densities [A.m-3]": 0,
                f"Sum of x-averaged {domain} electrode {phase_name}volumetric interfacial current densities [A.m-3]": 0,
            }
        )

        for reaction_name in reaction_names:
            variable_name = (
                f"{Domain} electrode {reaction_name}{phase_name}"
                "volumetric interfacial current density [A.m-3]"
            )
            if variable_name not in new_variables:
                continue
            a_j_k = new_variables[variable_name]

            if reaction_name == "":
                s_k = self.param.domain_params[domain].prim.s_plus_S
                electrolyte_source_term = s_k * a_j_k
            else:
                electrolyte_source_term = self._get_side_reaction_electrolyte_source_term(
                    new_variables,
                    domain,
                    reaction_name,
                    a_j_k,
                )

            new_variables[
                f"Sum of {domain} electrode {phase_name}electrolyte reaction source terms [A.m-3]"
            ] += electrolyte_source_term
            new_variables[
                f"Sum of x-averaged {domain} electrode {phase_name}electrolyte reaction source terms [A.m-3]"
            ] += pybamm.x_average(electrolyte_source_term)
            new_variables[
                f"Sum of {domain} electrode {phase_name}volumetric interfacial current densities [A.m-3]"
            ] += a_j_k
            new_variables[
                f"Sum of x-averaged {domain} electrode {phase_name}volumetric interfacial current densities [A.m-3]"
            ] += pybamm.x_average(a_j_k)

        variables.update(new_variables)
        return variables

from __future__ import annotations

import numpy as np

import pybamm
from pybamm.models.submodels.electrolyte_conductivity.surface_potential_form.full_surface_form_conductivity import (
    FullDifferential as SurfaceFormFullDifferential,
)
from pybamm.models.submodels.electrolyte_diffusion.base_electrolyte_diffusion import (
    BaseElectrolyteDiffusion,
)

from .directionality import directional_function_parameter
COMSOL_FORM_CONCENTRATION_FLOOR = pybamm.Scalar(100.0)
NONNEGATIVE_CONCENTRATION_FLOOR = pybamm.Scalar(0)


def _safe_concentration(c_e: pybamm.Symbol) -> pybamm.Symbol:
    """Mirror COMSOL's repeated `max(eps^2, c)` guards in transport terms."""

    return pybamm.maximum(c_e, COMSOL_FORM_CONCENTRATION_FLOOR)


def _nonnegative_concentration(c_e: pybamm.Symbol) -> pybamm.Symbol:
    """Exclude nonphysical negative concentrations from constitutive/source closures."""

    return pybamm.maximum(c_e, NONNEGATIVE_CONCENTRATION_FLOOR)


def _reaction_source(
    param: pybamm.LeadAcidParameters,
    a_j: pybamm.Symbol,
    c_e: pybamm.Symbol,
    t_plus: pybamm.Symbol,
    *,
    nu_hplus: float,
    nu_hso4: float,
    nu_h2o: float,
    electrons: float,
) -> pybamm.Symbol:
    """
    Build the COMSOL electrochemical volume source for one interfacial reaction.
    """

    coefficient = (
        (1 - c_e * param.V_e) * ((1 - t_plus) * nu_hplus + t_plus * nu_hso4)
        - c_e * param.V_w * nu_h2o
    )
    return -a_j * coefficient / (param.F * electrons)


class ComsolFormElectrolyteDiffusion(BaseElectrolyteDiffusion):
    """
    Electrolyte mass balance with `c_l` as the state, matching COMSOL Model1.
    """

    def __init__(self, param, options=None):
        super().__init__(param, options)

    def get_fundamental_variables(self):
        c_e_dict: dict[str, pybamm.Symbol] = {}
        for domain in self.options.whole_cell_domains:
            domain_name = domain.split()[0].capitalize()
            c_e_k = pybamm.Variable(
                f"{domain_name} electrolyte concentration [mol.m-3]",
                domain=domain,
                auxiliary_domains={"secondary": "current collector"},
                bounds=(0, np.inf),
                scale=self.param.c_e_init_av,
            )
            c_e_k.print_name = f"c_e_{domain[0]}"
            c_e_dict[domain] = c_e_k

        variables = self._get_standard_concentration_variables(c_e_dict)
        variables["Electrolyte concentration concatenation [mol.m-3]"] = variables[
            "Electrolyte concentration [mol.m-3]"
        ]
        return variables

    def _get_velocity(self, variables: dict[str, pybamm.Symbol]) -> pybamm.Symbol:
        """
        Recover COMSOL's through-cell velocity directly from the electrolyte current.
        """

        c_e_n = variables["Negative electrolyte concentration [mol.m-3]"]
        c_e_s = variables["Separator electrolyte concentration [mol.m-3]"]
        c_e_p = variables["Positive electrolyte concentration [mol.m-3]"]
        T_n = variables["Negative electrode temperature [K]"]
        T_s = variables["Separator temperature [K]"]
        T_p = variables["Positive electrode temperature [K]"]
        i_e = variables["Electrolyte current density [A.m-2]"]

        t_n = self.param.t_plus(c_e_n, T_n)
        t_s = self.param.t_plus(c_e_s, T_s)
        t_p = self.param.t_plus(c_e_p, T_p)

        coeff_n = (
            -0.5
            * (self.param.V_Pb - self.param.V_PbSO4 - (1 - 2 * t_n) * self.param.V_e)
            / self.param.F
        )
        coeff_s = (
            -0.5
            * (self.param.V_Pb - self.param.V_PbSO4 - (1 - 2 * t_s) * self.param.V_e)
            / self.param.F
        )
        coeff_p = (
            -0.5
            * (
                self.param.V_PbSO4
                - self.param.V_PbO2
                + 2 * self.param.V_w
                - (3 - 2 * t_p) * self.param.V_e
            )
            / self.param.F
        )

        coefficient = pybamm.concatenation(coeff_n, coeff_s, coeff_p)
        return coefficient * i_e

    def get_coupled_variables(self, variables):
        c_e = variables["Electrolyte concentration [mol.m-3]"]
        c_e_safe = _nonnegative_concentration(c_e)
        tor = variables["Electrolyte transport efficiency"]
        T = variables["Cell temperature [K]"]
        velocity = self._get_velocity(variables)

        diffusion_flux = -tor * self.param.D_e(c_e, T) * pybamm.grad(c_e)
        convection_flux = c_e_safe * velocity
        flux = diffusion_flux + convection_flux

        eps_c_e_dict = {}
        for domain in self.options.whole_cell_domains:
            porosity_name = domain.capitalize()
            concentration_name = domain.split()[0].capitalize()
            eps_k = variables[f"{porosity_name} porosity"]
            c_e_k = _nonnegative_concentration(
                variables[f"{concentration_name} electrolyte concentration [mol.m-3]"]
            )
            eps_c_e_dict[domain] = eps_k * c_e_k

        variables = variables.copy()
        variables.update(self._get_standard_porosity_times_concentration_variables(eps_c_e_dict))
        variables["COMSOL through-cell convection velocity [m.s-1]"] = velocity
        variables.update(self._get_standard_flux_variables(flux))
        variables.update(
            {
                "Electrolyte diffusion flux [mol.m-2.s-1]": diffusion_flux,
                "Electrolyte migration flux [mol.m-2.s-1]": pybamm.zeros_like(
                    diffusion_flux
                ),
                "Electrolyte convection flux [mol.m-2.s-1]": convection_flux,
            }
        )
        return variables

    def set_rhs(self, variables):
        c_e = variables["Electrolyte concentration [mol.m-3]"]
        c_e_n = variables["Negative electrolyte concentration [mol.m-3]"]
        c_e_n_safe = _nonnegative_concentration(c_e_n)
        c_e_s_safe = _nonnegative_concentration(
            variables["Separator electrolyte concentration [mol.m-3]"]
        )
        c_e_p = variables["Positive electrolyte concentration [mol.m-3]"]
        c_e_p_safe = _nonnegative_concentration(c_e_p)
        T = variables["Cell temperature [K]"]
        T_n = variables["Negative electrode temperature [K]"]
        T_p = variables["Positive electrode temperature [K]"]
        eps = variables["Porosity"]
        tor = variables["Electrolyte transport efficiency"]
        velocity = variables["COMSOL through-cell convection velocity [m.s-1]"]

        diffusion_term = pybamm.div(tor * self.param.D_e(c_e, T) * pybamm.grad(c_e))
        advection_term = pybamm.inner(velocity, pybamm.grad(c_e))

        t_n = self.param.t_plus(c_e_n, T_n)
        t_p = self.param.t_plus(c_e_p, T_p)

        a_j_n_main = variables[
            "Negative electrode volumetric interfacial current density [A.m-3]"
        ]
        a_j_n_hy = variables[
            "Negative electrode hydrogen volumetric interfacial current density [A.m-3]"
        ]
        a_j_p_main = variables[
            "Positive electrode volumetric interfacial current density [A.m-3]"
        ]
        a_j_p_ox = variables[
            "Positive electrode oxygen volumetric interfacial current density [A.m-3]"
        ]

        source_n_main = -_reaction_source(
            self.param,
            a_j_n_main,
            c_e_n_safe,
            t_n,
            nu_hplus=1.0,
            nu_hso4=-1.0,
            nu_h2o=0.0,
            electrons=2.0,
        )
        source_n = source_n_main + _reaction_source(
            self.param,
            a_j_n_hy,
            c_e_n_safe,
            t_n,
            nu_hplus=-2.0,
            nu_hso4=0.0,
            nu_h2o=0.0,
            electrons=2.0,
        )
        separator_source_rate = pybamm.Parameter(
            "Separator electrolyte source relaxation rate [s-1]"
        )
        source_s = separator_source_rate * (
            self.param.c_e_init - c_e_s_safe
        )
        source_p = _reaction_source(
            self.param,
            a_j_p_main,
            c_e_p_safe,
            t_p,
            nu_hplus=-3.0,
            nu_hso4=-1.0,
            nu_h2o=2.0,
            electrons=2.0,
        ) + _reaction_source(
            self.param,
            a_j_p_ox,
            c_e_p_safe,
            t_p,
            nu_hplus=-2.0,
            nu_hso4=0.0,
            nu_h2o=1.0,
            electrons=2.0,
        )

        source = pybamm.concatenation(source_n, source_s, source_p)
        self.rhs = {c_e: (diffusion_term - advection_term + source) / eps}

    def set_initial_conditions(self, variables):
        c_e = variables["Electrolyte concentration [mol.m-3]"]
        self.initial_conditions = {c_e: self.param.c_e_init}

    def set_boundary_conditions(self, variables):
        c_e = variables["Electrolyte concentration [mol.m-3]"]
        zero = pybamm.Scalar(0)
        self.boundary_conditions = {
            c_e: {"left": (zero, "Neumann"), "right": (zero, "Neumann")}
        }


class ComsolFormSurfaceFormDifferential(SurfaceFormFullDifferential):
    """Surface-form conductivity that uses COMSOL's liquid-current law."""

    def _initial_surface_potential_difference(self) -> pybamm.Symbol:
        Domain = self.domain.capitalize()
        return directional_function_parameter(
            pybamm.Scalar(0),
            charge_name=f"Charge {Domain} electrode open-circuit potential [V]",
            discharge_name=f"Discharge {Domain} electrode open-circuit potential [V]",
            inputs={"Electrolyte molar mass [mol.kg-1]": self.param.m(self.param.c_e_init)},
        )

    def get_fundamental_variables(self):
        if self.domain == "separator":
            return {}

        domain, Domain = self.domain_Domain
        delta_phi = pybamm.Variable(
            f"{Domain} electrode surface potential difference [V]",
            domain=f"{domain} electrode",
            auxiliary_domains={"secondary": "current collector"},
            reference=self._initial_surface_potential_difference(),
        )
        variables = self._get_standard_average_surface_potential_difference_variables(
            pybamm.x_average(delta_phi)
        )
        variables.update(
            self._get_standard_surface_potential_difference_variables(delta_phi)
        )
        return variables

    def _concentration_coefficient(
        self,
        c_e: pybamm.Symbol,
        temperature_k: pybamm.Symbol,
    ) -> pybamm.Symbol:
        c_e_safe = _safe_concentration(c_e)
        return (
            self.param.R
            * temperature_k
            / self.param.F
            * (1 - 2 * self.param.t_plus(c_e_safe, temperature_k))
            / c_e_safe
        )

    def _concentration_term(
        self,
        c_e: pybamm.Symbol,
        temperature_k: pybamm.Symbol,
    ) -> pybamm.Symbol:
        return self._concentration_coefficient(c_e, temperature_k) * pybamm.grad(c_e)

    def _get_comsol_overpotentials(self, variables):
        phi_e_n = variables["Negative electrolyte potential [V]"]
        phi_e_p = variables["Positive electrolyte potential [V]"]

        c_e_n = variables["Negative electrolyte concentration [mol.m-3]"]
        c_e_s = variables["Separator electrolyte concentration [mol.m-3]"]
        c_e_p = variables["Positive electrolyte concentration [mol.m-3]"]

        T_n = variables["Negative electrode temperature [K]"]
        T_s = variables["Separator temperature [K]"]
        T_p = variables["Positive electrode temperature [K]"]

        integral_n = pybamm.IndefiniteIntegral(
            self._concentration_term(c_e_n, T_n),
            pybamm.standard_spatial_vars.x_n,
        )
        integral_s_local = pybamm.IndefiniteIntegral(
            self._concentration_term(c_e_s, T_s),
            pybamm.standard_spatial_vars.x_s,
        )
        integral_s = integral_s_local + pybamm.boundary_value(integral_n, "right")
        integral_p_local = pybamm.IndefiniteIntegral(
            self._concentration_term(c_e_p, T_p),
            pybamm.standard_spatial_vars.x_p,
        )
        integral_p = integral_p_local + pybamm.boundary_value(integral_s, "right")

        eta_c_av = pybamm.x_average(integral_p) - pybamm.x_average(integral_n)
        delta_phi_e_av = (
            pybamm.x_average(phi_e_p) - pybamm.x_average(phi_e_n) - eta_c_av
        )

        variables.update(self._get_split_overpotential(eta_c_av, delta_phi_e_av))
        return variables

    def get_coupled_variables(self, variables):
        domain_name = self.domain.capitalize()

        if self.domain in ["negative", "positive"]:
            conductivity, sigma_eff = self._get_conductivities(variables)
            i_boundary_cc = variables["Current collector current density [A.m-2]"]
            c_e = variables[f"{domain_name} electrolyte concentration [mol.m-3]"]
            delta_phi = variables[
                f"{domain_name} electrode surface potential difference [V]"
            ]
            T = variables[f"{domain_name} electrode temperature [K]"]

            i_e = conductivity * (
                self._concentration_term(c_e, T)
                + pybamm.grad(delta_phi)
                + i_boundary_cc / sigma_eff
            )
            variables[f"{domain_name} electrolyte current density [A.m-2]"] = i_e

            phi_s = variables[f"{domain_name} electrode potential [V]"]
            phi_e = phi_s - delta_phi

        elif self.domain == "separator":
            x_s = pybamm.standard_spatial_vars.x_s
            i_boundary_cc = variables["Current collector current density [A.m-2]"]
            c_e_s = variables["Separator electrolyte concentration [mol.m-3]"]

            if self.options.electrode_types["negative"] == "planar":
                phi_e_n_s = variables["Lithium metal interface electrolyte potential [V]"]
            else:
                phi_e_n = variables["Negative electrolyte potential [V]"]
                phi_e_n_s = pybamm.boundary_value(phi_e_n, "right")

            tor_s = variables["Separator electrolyte transport efficiency"]
            T = variables["Separator temperature [K]"]
            kappa_s_eff = self.param.kappa_e(c_e_s, T) * tor_s

            phi_e = phi_e_n_s + pybamm.IndefiniteIntegral(
                self._concentration_term(c_e_s, T) - i_boundary_cc / kappa_s_eff,
                x_s,
            )

            i_e = pybamm.PrimaryBroadcastToEdges(i_boundary_cc, "separator")
            variables[f"{domain_name} electrolyte current density [A.m-2]"] = i_e

            self.boundary_conditions[c_e_s] = {
                "left": (pybamm.boundary_gradient(c_e_s, "left"), "Neumann"),
                "right": (pybamm.boundary_gradient(c_e_s, "right"), "Neumann"),
            }

        variables[f"{domain_name} electrolyte potential [V]"] = phi_e

        if self.domain == "positive":
            phi_e_dict = {}
            i_e_dict = {}
            for domain in self.options.whole_cell_domains:
                subdomain_name = domain.capitalize().split()[0]
                phi_e_dict[domain] = variables[f"{subdomain_name} electrolyte potential [V]"]
                i_e_dict[domain] = variables[
                    f"{subdomain_name} electrolyte current density [A.m-2]"
                ]

            variables.update(self._get_standard_potential_variables(phi_e_dict))
            i_e = pybamm.concatenation(*i_e_dict.values())
            variables.update(self._get_standard_current_variables(i_e))
            variables = self._get_comsol_overpotentials(variables)

        if self.domain == "negative":
            grad_c_e = pybamm.boundary_gradient(c_e, "right")
            coeff_right = pybamm.boundary_value(
                self._concentration_coefficient(c_e, T),
                "right",
            )
            grad_left = -i_boundary_cc * pybamm.boundary_value(1 / sigma_eff, "left")
            grad_right = (
                (i_boundary_cc / pybamm.boundary_value(conductivity, "right"))
                - coeff_right * grad_c_e
                - i_boundary_cc * pybamm.boundary_value(1 / sigma_eff, "right")
            )
        elif self.domain == "positive":
            grad_c_e = pybamm.boundary_gradient(c_e, "left")
            coeff_left = pybamm.boundary_value(
                self._concentration_coefficient(c_e, T),
                "left",
            )
            grad_left = (
                (i_boundary_cc / pybamm.boundary_value(conductivity, "left"))
                - coeff_left * grad_c_e
                - i_boundary_cc * pybamm.boundary_value(1 / sigma_eff, "left")
            )
            grad_right = -i_boundary_cc * pybamm.boundary_value(1 / sigma_eff, "right")

        if self.domain in ["negative", "positive"]:
            variables.update(
                {
                    f"{self.domain} grad(delta_phi) left": grad_left,
                    f"{self.domain} grad(delta_phi) right": grad_right,
                    f"{self.domain} grad(c_e) internal": grad_c_e,
                }
            )
        return variables

    def set_rhs(self, variables):
        if self.domain == "separator":
            return

        domain, Domain = self.domain_Domain
        T = variables[f"{Domain} electrode temperature [K]"]
        C_dl = self.domain_param.C_dl(T)
        delta_phi = variables[f"{Domain} electrode surface potential difference [V]"]
        i_e = variables[f"{Domain} electrolyte current density [A.m-2]"]
        sum_a_j = variables[
            f"Sum of {domain} electrode volumetric interfacial current densities [A.m-3]"
        ]
        area = variables[f"{Domain} electrode surface area to volume ratio [m-1]"]
        self.rhs[delta_phi] = 1 / (area * C_dl) * (pybamm.div(i_e) - sum_a_j)

    def set_initial_conditions(self, variables):
        if self.domain == "separator":
            return

        Domain = self.domain.capitalize()
        delta_phi = variables[f"{Domain} electrode surface potential difference [V]"]
        self.initial_conditions = {delta_phi: self._initial_surface_potential_difference()}

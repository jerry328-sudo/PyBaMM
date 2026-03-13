from __future__ import annotations

import pybamm

from pybamm.input.parameters.lead_acid.COMSOLModel1 import get_parameter_values

from ..full import Full


DEFAULT_OPTIONS: dict[str, str] = {
    "convection": "none",
    "hydrolysis": "false",
    "intercalation kinetics": "asymmetric Butler-Volmer",
    "surface form": "differential",
}


class COMSOLModel1(Full):
    """
    Lead-acid model with PDEs organised to mirror the exported COMSOL weak forms.

    The standard PyBaMM lead-acid models remain untouched. This class adds a new,
    parallel model line that reuses PyBaMM's infrastructure but swaps in the
    COMSOL-specific electrolyte, kinetics and source-term closures.
    """

    def __init__(self, options: dict[str, str] | None = None, *, build: bool = True) -> None:
        merged_options = dict(DEFAULT_OPTIONS)
        if options:
            merged_options.update(options)
        super().__init__(
            options=merged_options,
            name="COMSOL Model1 lead-acid model",
            build=build,
        )

    @property
    def default_parameter_values(self) -> pybamm.ParameterValues:
        return pybamm.ParameterValues(get_parameter_values())

    @property
    def default_var_pts(self) -> dict[str, int]:
        """
        Use a denser through-cell mesh than the stock lead-acid default.

        The COMSOL benchmark is more advection-sensitive than the stock PyBaMM
        lead-acid examples. A moderately finer mesh reduces the positive-electrode
        concentration undershoot without changing the model equations.
        """

        return {"x_n": 80, "x_s": 140, "x_p": 100, "y": 10, "z": 10}

    def set_porosity_submodel(self) -> None:
        from .submodels.porosity import ComsolReactionDrivenPorosity

        self.submodels["porosity"] = ComsolReactionDrivenPorosity(
            self.param,
            self.options,
            False,
        )

    def set_open_circuit_potential_submodel(self) -> None:
        from .submodels.open_circuit_potential import (
            ComsolConstantSideReactionOpenCircuitPotential,
            ComsolMainReactionOpenCircuitPotential,
        )

        for domain in ["negative", "positive"]:
            self.submodels[f"{domain} open-circuit potential"] = (
                ComsolMainReactionOpenCircuitPotential(
                    self.param,
                    domain,
                    "lead-acid main",
                    self.options,
                    "primary",
                )
            )

        self.submodels["positive oxygen open-circuit potential"] = (
            ComsolConstantSideReactionOpenCircuitPotential(
                self.param,
                "positive",
                reaction="lead-acid oxygen",
                reaction_name="oxygen ",
                potential_parameter_name="Oxygen reference OCP vs SHE [V]",
                options=self.options,
            )
        )
        self.submodels["negative hydrogen open-circuit potential"] = (
            ComsolConstantSideReactionOpenCircuitPotential(
                self.param,
                "negative",
                reaction="lead-acid hydrogen",
                reaction_name="hydrogen ",
                potential_parameter_name="Hydrogen reference OCP vs SHE [V]",
                options=self.options,
            )
        )

    def set_intercalation_kinetics_submodel(self) -> None:
        from .submodels.kinetics import ComsolMainReactionKinetics

        for domain in ["negative", "positive"]:
            self.submodels[f"{domain} interface"] = ComsolMainReactionKinetics(
                self.param,
                domain,
                "lead-acid main",
                self.options,
                "primary",
            )

    def set_transport_efficiency_submodels(self) -> None:
        from .submodels.transport_efficiency import ComsolTransportEfficiency

        self.submodels["electrolyte transport efficiency"] = ComsolTransportEfficiency(
            self.param,
            "Electrolyte",
            self.options,
        )
        self.submodels["electrode transport efficiency"] = ComsolTransportEfficiency(
            self.param,
            "Electrode",
            self.options,
        )

    def set_electrolyte_submodel(self) -> None:
        from .submodels.electrolyte import (
            ComsolFormElectrolyteDiffusion,
            ComsolFormSurfaceFormDifferential,
        )

        self.submodels["electrolyte diffusion"] = ComsolFormElectrolyteDiffusion(
            self.param,
            self.options,
        )

        for domain in ["negative", "separator", "positive"]:
            self.submodels[f"{domain} surface potential difference"] = (
                ComsolFormSurfaceFormDifferential(
                    self.param,
                    domain,
                    self.options,
                )
            )

    def set_side_reaction_submodels(self) -> None:
        from .submodels.kinetics import (
            ComsolNegativeHydrogenKinetics,
            ComsolPositiveOxygenKinetics,
        )

        self.submodels["oxygen diffusion"] = pybamm.oxygen_diffusion.NoOxygen(self.param)
        self.submodels["positive oxygen interface"] = ComsolPositiveOxygenKinetics(
            self.param,
            "positive",
            "lead-acid oxygen",
            self.options,
            "primary",
        )
        self.submodels["negative oxygen interface"] = pybamm.kinetics.NoReaction(
            self.param,
            "negative",
            "lead-acid oxygen",
            self.options,
            "primary",
        )
        self.submodels["negative hydrogen interface"] = ComsolNegativeHydrogenKinetics(
            self.param,
            "negative",
            self.options,
            "primary",
        )

    def set_total_interface_submodel(self) -> None:
        from .submodels.total_interfacial_current import (
            ComsolLeadAcidTotalInterfacialCurrent,
        )

        self.submodels["total interface"] = ComsolLeadAcidTotalInterfacialCurrent(
            self.param,
            self.options,
        )


class COMSOLCurrentDistributionInitialization(COMSOLModel1):
    """
    Stationary current-distribution initialiser mirroring COMSOL's CDI study.

    This model freezes porosity and electrolyte concentration at their initial
    fields, disables all Faradaic reactions, and solves only for the charge
    distribution under the applied boundary current with

        phi_s - phi_e = U_eq

    enforced locally. Its solution is then used to seed the transient
    `COMSOLModel1` run without relying on exported COMSOL field data.
    """

    def __init__(self, options: dict[str, str] | None = None, *, build: bool = True) -> None:
        merged_options = dict(DEFAULT_OPTIONS)
        if options:
            merged_options.update(options)
        Full.__init__(
            self,
            options=merged_options,
            name="COMSOL Model1 current-distribution initialization",
            build=build,
        )

    def set_external_circuit_submodel(self) -> None:
        from .submodels.current_distribution import ComsolZeroThroughput

        self.submodels["external circuit"] = pybamm.external_circuit.ExplicitCurrentControl(
            self.param,
            self.options,
        )
        self.submodels["discharge and throughput variables"] = ComsolZeroThroughput(
            self.param,
            self.options,
        )

    def set_porosity_submodel(self) -> None:
        from .submodels.current_distribution import (
            ComsolCurrentDistributionConstantPorosity,
        )

        self.submodels["porosity"] = ComsolCurrentDistributionConstantPorosity(
            self.param,
            self.options,
        )

    def set_intercalation_kinetics_submodel(self) -> None:
        from .submodels.current_distribution import ComsolMainReactionInitialization

        for domain in ["negative", "positive"]:
            self.submodels[f"{domain} interface"] = ComsolMainReactionInitialization(
                self.param,
                domain,
                "lead-acid main",
                self.options,
                "primary",
            )

    def set_electrolyte_submodel(self) -> None:
        from .submodels.current_distribution import (
            ComsolCurrentDistributionConstantConcentration,
            ComsolCurrentDistributionInitializationSurfaceForm,
        )

        self.submodels["electrolyte diffusion"] = (
            ComsolCurrentDistributionConstantConcentration(
                self.param,
                self.options,
            )
        )

        for domain in ["negative", "separator", "positive"]:
            self.submodels[f"{domain} surface potential difference"] = (
                ComsolCurrentDistributionInitializationSurfaceForm(
                    self.param,
                    domain,
                    self.options,
                )
            )

    def set_side_reaction_submodels(self) -> None:
        from .submodels.current_distribution import ComsolNegativeHydrogenNoReaction

        self.submodels["oxygen diffusion"] = pybamm.oxygen_diffusion.NoOxygen(self.param)
        self.submodels["positive oxygen interface"] = pybamm.kinetics.NoReaction(
            self.param,
            "positive",
            "lead-acid oxygen",
            self.options,
            "primary",
        )
        self.submodels["negative oxygen interface"] = pybamm.kinetics.NoReaction(
            self.param,
            "negative",
            "lead-acid oxygen",
            self.options,
            "primary",
        )
        self.submodels["negative hydrogen interface"] = ComsolNegativeHydrogenNoReaction(
            self.param,
            "negative",
            self.options,
            "primary",
        )

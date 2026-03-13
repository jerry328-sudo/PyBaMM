from __future__ import annotations

import pybamm

from pybamm.models.submodels.transport_efficiency.base_transport_efficiency import BaseModel


class ComsolTransportEfficiency(BaseModel):
    """
    Transport-efficiency closure matching the COMSOL lead-acid interface inputs.

    The electrolyte branch keeps the standard `epsilon^b_e` law. The electrode
    branch intentionally uses `epsilon^b_s` instead of PyBaMM's default
    `(1 - epsilon)^b_s`, because that is the algebra exported by COMSOL Model1.
    """

    def __init__(self, param, component: str, options=None):
        super().__init__(param, component, options=options)

    def get_coupled_variables(self, variables):
        tor_dict = {}

        if self.component == "Electrolyte":
            for domain in self.options.whole_cell_domains:
                Domain = domain.capitalize()
                epsilon = variables[f"{Domain} porosity"]
                exponent = self.param.domain_params[domain.split()[0]].b_e
                tor_dict[domain] = epsilon**exponent
        elif self.component == "Electrode":
            for domain in self.options.whole_cell_domains:
                if domain == "separator":
                    tor_dict[domain] = pybamm.FullBroadcast(0, "separator", "current collector")
                    continue

                Domain = domain.capitalize()
                epsilon = variables[f"{Domain} porosity"]
                exponent = self.param.domain_params[domain.split()[0]].b_s
                tor_dict[domain] = epsilon**exponent
        else:
            raise ValueError(f"Unsupported transport-efficiency component {self.component!r}")

        variables.update(self._get_standard_transport_efficiency_variables(tor_dict))
        return variables

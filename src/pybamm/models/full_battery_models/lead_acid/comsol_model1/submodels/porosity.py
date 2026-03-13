from __future__ import annotations

import pybamm
from pybamm.models.submodels.porosity.reaction_driven_porosity_ode import ReactionDrivenODE


class ComsolReactionDrivenPorosity(ReactionDrivenODE):
    """Porosity ODE with explicit initial porosity controls for all three domains."""

    def set_initial_conditions(self, variables: dict[str, pybamm.Symbol]) -> None:
        if self.x_average is True:
            for domain in self.options.whole_cell_domains:
                eps_av = variables[f"X-averaged {domain} porosity"]
                self.initial_conditions[eps_av] = pybamm.Parameter(
                    f"Initial porosity of {domain}"
                )
            return

        eps = variables["Porosity"]
        eps_init = pybamm.concatenation(
            pybamm.FullBroadcast(
                pybamm.Parameter("Initial porosity of negative electrode"),
                "negative electrode",
                "current collector",
            ),
            pybamm.FullBroadcast(
                pybamm.Parameter("Initial porosity of separator"),
                "separator",
                "current collector",
            ),
            pybamm.FullBroadcast(
                pybamm.Parameter("Initial porosity of positive electrode"),
                "positive electrode",
                "current collector",
            ),
        )
        self.initial_conditions = {eps: eps_init}

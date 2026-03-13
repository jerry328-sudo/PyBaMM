# COMSOL Model1 material tables

This directory stores the COMSOL-exported 1D material property tables used by
`pybamm.input.parameters.lead_acid.COMSOLModel1`.

The numeric tables use the standard PyBaMM parameter-data style:
- CSV format
- one header row
- two columns: `concentration_mol_m3`, `value`

Files:
- `sulfuric_acid_conductivity_sigmal_int1(c).csv`
- `sulfuric_acid_diffusivity_Dl_int1(c).csv`
- `pb_equilibrium_potential_Eeq_Pb_int1(c).csv`
- `pbo2_equilibrium_potential_Eeq_PbO2_int1(c).csv`

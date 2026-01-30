"""
Hamiltonian Generation Module

This module provides tools to generate quantum Hamiltonians for:
- Lattice models (e.g., Transverse Field Ising Model)
- Molecular systems (using PySCF and OpenFermion)
"""

from .lattice import two_dimensional_transverse_field_ising_model
from .molecular import make_molecular_hamil
from .geometry import (
    get_H2_geo, get_H4_geo, get_H6_geo,
    get_LiH_geo, get_N2_geo, get_H2O_geo,
    get_NH3_geo, get_C2H2_geo, get_C2H4_geo, get_CO2_geo,
    get_CH3OH_geo, get_C2H5OH_geo, get_benzene_geo,
    equilibrium_geometry_dict, geometry_generator_dict,
    molecule_qubit_counts
)
from .save import save_hamil

__all__ = [
    'two_dimensional_transverse_field_ising_model',
    'make_molecular_hamil',
    'save_hamil',
    'get_H2_geo', 'get_H4_geo', 'get_H6_geo',
    'get_LiH_geo', 'get_N2_geo', 'get_H2O_geo',
    'get_NH3_geo', 'get_C2H2_geo', 'get_C2H4_geo', 'get_CO2_geo',
    'get_CH3OH_geo', 'get_C2H5OH_geo', 'get_benzene_geo',
    'equilibrium_geometry_dict', 'geometry_generator_dict',
    'molecule_qubit_counts'
]

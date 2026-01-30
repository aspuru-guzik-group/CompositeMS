# Hamiltonian Generator Module

This module provides tools to generate quantum Hamiltonians for various systems.

## Features

### 1. Lattice Models
Generate Hamiltonians for lattice systems without any external dependencies.

**Example: Transverse Field Ising Model (TFIM)**
```python
from composite_ms.hamil_generator import two_dimensional_transverse_field_ising_model

# Generate a 4x4 TFIM with transverse field strength g=1.0
hamil = two_dimensional_transverse_field_ising_model(n_col=4, n_row=4, g=1.0)
print(f"Generated TFIM with {len(hamil.terms)} terms")
```

### 2. Molecular Systems
Generate Hamiltonians for molecular systems using quantum chemistry calculations.

**Requirements:**
```bash
pip install openfermion openfermionpyscf pyscf
```

**Example: Generate H2O Hamiltonian**
```python
from composite_ms.hamil_generator import make_molecular_hamil, get_H2O_geo
from openfermion.transforms import jordan_wigner

# Get H2O geometry at equilibrium
geometry = get_H2O_geo(bond_len=0.96)

# Generate Hamiltonian using Jordan-Wigner transformation
qubit_hamil, fermion_hamil, energies = make_molecular_hamil(
    geometry,
    basis="sto-3g",
    run_fci=True,
    fermi_qubit_transform=jordan_wigner
)

print(f"Hartree-Fock energy: {energies['hf']:.6f} Ha")
print(f"FCI energy: {energies['fci']:.6f} Ha")
```

### 3. Pre-defined Molecular Geometries
The module includes geometry generators for common molecules:

```python
from composite_ms.hamil_generator import (
    get_H2_geo, get_H4_geo, get_H6_geo,
    get_LiH_geo, get_N2_geo, get_H2O_geo,
    get_NH3_geo, get_C2H2_geo, get_C2H4_geo, get_CO2_geo,
    equilibrium_geometry_dict
)

# Get equilibrium bond length for H2
bond_len = equilibrium_geometry_dict["H2"]  # 0.74 Angstroms

# Generate geometry
geometry = get_H2_geo(bond_len)
```

### 4. Saving Generated Hamiltonians
Save your generated Hamiltonians to the standard format:

```python
from composite_ms.hamil_generator import save_hamil

# After generating a Hamiltonian
save_hamil(
    hamil=qubit_hamil,
    n_qubit=10,
    category="mol",  # or "lattice"
    title="my_custom_hamiltonian",
    other_info={"energy": -1.234, "basis": "sto-3g"}
)
```

This saves to `composite_ms/hamil/{category}/{title}.op` by default.

## Available Transformations

For molecular Hamiltonians, you can use different fermion-to-qubit transformations:

```python
from openfermion.transforms import jordan_wigner, bravyi_kitaev

# Jordan-Wigner
hamil_jw, _, _ = make_molecular_hamil(geometry, fermi_qubit_transform=jordan_wigner)

# Bravyi-Kitaev (often more efficient, recommended)
hamil_bk, _, _ = make_molecular_hamil(geometry, fermi_qubit_transform=bravyi_kitaev)
```

**Note**: The Parity transformation may also be available depending on your OpenFermion version.

## Module Structure

```
composite_ms/hamil_generator/
├── __init__.py         # Main exports
├── lattice.py          # Lattice model generators (TFIM, etc.)
├── molecular.py        # Molecular Hamiltonian generator
├── geometry.py         # Molecular geometry definitions
├── save.py             # Save Hamiltonians to disk
└── README.md           # This file
```

## Complete Example

See `example_generate_hamil.py` in the repository root for complete working examples.

## Notes

- Lattice generation works without any extra dependencies
- Molecular generation requires `openfermion`, `openfermionpyscf`, and `pyscf`
- All pre-generated Hamiltonians are already included in `composite_ms/hamil/`
- You only need this module if you want to generate new custom Hamiltonians

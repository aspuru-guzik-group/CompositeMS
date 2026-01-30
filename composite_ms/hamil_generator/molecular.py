"""
Molecular Hamiltonian Generators

Functions to generate Hamiltonians for molecular systems using PySCF and OpenFermion.
"""

try:
    from openfermion.transforms import bravyi_kitaev, get_fermion_operator
    from openfermionpyscf import run_pyscf
    from openfermion.chem import MolecularData
    OPENFERMION_AVAILABLE = True
except ImportError:
    OPENFERMION_AVAILABLE = False

CHEMICAL_ACCURACY = 0.001


def make_molecular_hamil(geometry, basis="sto-3g", run_fci=False, fermi_qubit_transform=None):
    """
    Generate a molecular Hamiltonian using PySCF and OpenFermion.
    
    Args:
        geometry: List of tuples (atom, (x, y, z)) specifying molecular geometry
        basis: Basis set for quantum chemistry calculation (default: "sto-3g")
        run_fci: Whether to run Full Configuration Interaction (default: False)
        fermi_qubit_transform: Fermion-to-qubit transformation function
                              (default: bravyi_kitaev)
    
    Returns:
        tuple: (qubit_hamiltonian, fermion_hamiltonian, energy_dict)
               where energy_dict contains 'hf' (Hartree-Fock) and 'fci' energies
    
    Raises:
        ImportError: If openfermion or openfermionpyscf are not installed
    """
    if not OPENFERMION_AVAILABLE:
        raise ImportError(
            "OpenFermion and OpenFermion-PySCF are required to generate molecular Hamiltonians.\n"
            "Install with: pip install openfermion openfermionpyscf pyscf"
        )
    
    if fermi_qubit_transform is None:
        fermi_qubit_transform = bravyi_kitaev
    
    # Get fermion Hamiltonian
    multiplicity = 1
    charge = 0
    molecule = MolecularData(geometry, basis, multiplicity, charge)
    molecule.symmetry = True

    molecule = run_pyscf(molecule, run_fci=run_fci, verbose=False)

    fermion_hamiltonian = get_fermion_operator(molecule.get_molecular_hamiltonian())
    # Map fermion Hamiltonian to qubit Hamiltonian
    qubit_hamiltonian = fermi_qubit_transform(fermion_hamiltonian)
    # Ignore terms in Hamiltonian that close to zero
    qubit_hamiltonian.compress()

    return qubit_hamiltonian, fermion_hamiltonian, {"hf": molecule.hf_energy, "fci": molecule.fci_energy}

#!/usr/bin/env python
"""
Generate Large Molecular Hamiltonians (32+ qubits)

This script provides multiple options to generate Hamiltonians with at least 32 qubits.

Requirements:
    pip install openfermion openfermionpyscf pyscf
"""

from composite_ms.hamil_generator import make_molecular_hamil, save_hamil
from openfermion.transforms import jordan_wigner, bravyi_kitaev

# Try to import parity transform (optional)
try:
    from openfermion.transforms import parity
except ImportError:
    parity = None


def count_qubits(hamil):
    """Count the number of qubits in a Hamiltonian."""
    n_qubits = 0
    for term in hamil.terms.keys():
        if term:  # Not the identity term
            max_idx = max([idx for idx, _ in term])
            n_qubits = max(n_qubits, max_idx + 1)
    return n_qubits


def generate_molecule(name, geometry, basis="sto-3g", transform_name="BK", transform_func=bravyi_kitaev):
    """Generate a single molecular Hamiltonian."""
    print(f"\n{'='*70}")
    print(f"Generating {name} with {basis} basis and {transform_name} transformation")
    print(f"{'='*70}")
    
    print(f"\nMolecular geometry ({len(geometry)} atoms):")
    atom_counts = {}
    for atom, _ in geometry:
        atom_counts[atom] = atom_counts.get(atom, 0) + 1
    print(f"  Formula: {' '.join([f'{atom}{count}' if count > 1 else atom for atom, count in sorted(atom_counts.items())])}")
    
    print("\nRunning quantum chemistry calculation...")
    try:
        qubit_hamil, fermion_hamil, energies = make_molecular_hamil(
            geometry,
            basis=basis,
            run_fci=False,
            fermi_qubit_transform=transform_func
        )
        
        n_qubits = count_qubits(qubit_hamil)
        n_terms = len(qubit_hamil.terms)
        
        print(f"\n✓ Successfully generated {name}_{n_qubits}_{transform_name}")
        print(f"  Number of qubits: {n_qubits}")
        print(f"  Number of terms: {n_terms}")
        print(f"  Hartree-Fock energy: {energies['hf']:.6f} Ha")
        
        if n_qubits >= 32:
            print(f"  ✓ Target achieved: {n_qubits} >= 32 qubits")
        else:
            print(f"  ⚠ Warning: Only {n_qubits} qubits (target: 32+)")
            return None
        
        # Save the Hamiltonian
        filename = f"{name}_{n_qubits}_{transform_name}"
        save_hamil(
            qubit_hamil,
            n_qubits,
            "mol",
            filename,
            other_info={
                "hf_energy": energies['hf'],
                "basis": basis,
                "molecule": name,
                "transformation": transform_name,
                "n_atoms": len(geometry)
            }
        )
        
        return {
            'hamil': qubit_hamil,
            'n_qubits': n_qubits,
            'n_terms': n_terms,
            'energies': energies,
            'filename': filename
        }
        
    except Exception as e:
        print(f"✗ Failed to generate: {e}")
        return None


def main():
    print("="*70)
    print("Large Molecule Hamiltonian Generator (32+ qubits)")
    print("="*70)
    
    # Check dependencies
    try:
        import openfermion
        import openfermionpyscf
        import pyscf
    except ImportError as e:
        print("\n✗ Error: Missing required dependencies")
        print("Please install with:")
        print("  pip install openfermion openfermionpyscf pyscf")
        print(f"\nError details: {e}")
        return
    
    # Molecule options with guaranteed 32+ qubits
    molecules = []
    
    # Option 1: Methanol with 6-31g basis (should give ~38-48 qubits)
    molecules.append({
        'name': 'CH3OH',
        'description': 'Methanol with split-valence basis',
        'geometry': [
            ('C', (0.0416, 0.0000, 0.0000)),
            ('O', (1.4080, 0.0000, 0.0000)),
            ('H', (2.0000, 0.0000, 0.0000)),
            ('H', (-0.3584, -0.5150, 0.8900)),
            ('H', (-0.3584, -0.5150, -0.8900)),
            ('H', (-0.3584, 1.0300, 0.0000))
        ],
        'basis': '6-31g'
    })
    
    # Option 2: Ethanol (C2H5OH) with sto-3g (should give ~32-40 qubits)
    molecules.append({
        'name': 'C2H5OH',
        'description': 'Ethanol (9 atoms)',
        'geometry': [
            ('C', (-0.7479, 0.0564, 0.0000)),
            ('C', (0.7479, -0.0564, 0.0000)),
            ('O', (1.2479, -1.2000, 0.0000)),
            ('H', (-1.1479, 1.0564, 0.0000)),
            ('H', (-1.1479, -0.4436, 0.8900)),
            ('H', (-1.1479, -0.4436, -0.8900)),
            ('H', (1.1479, 0.4436, 0.8900)),
            ('H', (1.1479, 0.4436, -0.8900)),
            ('H', (2.2000, -1.2000, 0.0000))
        ],
        'basis': 'sto-3g'
    })
    
    # Option 3: H12 chain with sto-3g (should give ~24 qubits, backup)
    molecules.append({
        'name': 'H12',
        'description': 'Hydrogen chain (12 atoms)',
        'geometry': [('H', (i*1.0, 0.0, 0.0)) for i in range(12)],
        'basis': 'sto-3g'
    })
    
    # Option 4: H16 chain with sto-3g (should give ~32 qubits)
    molecules.append({
        'name': 'H16',
        'description': 'Hydrogen chain (16 atoms)',
        'geometry': [('H', (i*1.0, 0.0, 0.0)) for i in range(16)],
        'basis': 'sto-3g'
    })
    
    print("\nAvailable molecules to generate:")
    for i, mol in enumerate(molecules, 1):
        print(f"  {i}. {mol['name']} - {mol['description']} (basis: {mol['basis']})")
    
    # Generate all molecules that meet the criteria
    print("\n" + "="*70)
    print("Generating molecules...")
    print("="*70)
    
    results = []
    for mol in molecules:
        result = generate_molecule(
            mol['name'],
            mol['geometry'],
            basis=mol['basis'],
            transform_name='BK',
            transform_func=bravyi_kitaev
        )
        if result and result['n_qubits'] >= 32:
            results.append(result)
            print(f"\n✓ {mol['name']} meets requirements!")
            
            # If we got one with 32+ qubits, also generate with JW
            if result['n_qubits'] >= 32:
                print(f"\nAlso generating with Jordan-Wigner transformation...")
                result_jw = generate_molecule(
                    mol['name'],
                    mol['geometry'],
                    basis=mol['basis'],
                    transform_name='JW',
                    transform_func=jordan_wigner
                )
                if result_jw:
                    results.append(result_jw)
                
                # Stop after generating the first successful molecule
                break
    
    # Summary
    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    
    if results:
        print(f"\nSuccessfully generated {len(results)} Hamiltonian(s) with 32+ qubits:")
        print(f"\n{'Molecule':<15} {'Transform':<12} {'Qubits':<10} {'Terms':<12} {'File'}")
        print("-"*70)
        for result in results:
            mol_name = result['filename'].split('_')[0]
            transform = result['filename'].split('_')[-1]
            print(f"{mol_name:<15} {transform:<12} {result['n_qubits']:<10} "
                  f"{result['n_terms']:<12} {result['filename']}.op")
        
        print("\n" + "="*70)
        print("How to use these Hamiltonians:")
        print("="*70)
        print("\n```python")
        print("from composite_ms.hamil import get_test_hamil")
        print("")
        for result in results[:2]:  # Show first 2
            print(f"hamil = get_test_hamil('mol', '{result['filename']}')  # {result['n_qubits']} qubits")
        print("```")
        
    else:
        print("\n✗ No molecules met the 32+ qubit requirement.")
        print("Try using a larger basis set or a larger molecule.")
    
    print("\n✓ Generation complete!")


if __name__ == '__main__':
    main()

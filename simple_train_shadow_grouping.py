"""
Run Shadow Grouping method on the same problem as simple_train.py
This demonstrates the shadow grouping approach for comparison with CMS-LBCS
"""

from composite_ms.hamil import get_test_hamil
from composite_ms.other_methods.shadow_grouping import ShadowGroupingMeasurement
import numpy as np

def main():
    # Configuration - same as simple_train.py
    mol_name = "CO2_30_BK"  # Using CO2 molecule
    min_nshot_a_term = 10  # Minimum number of shots per term
    max_nshot = 20000  # Maximum number of shots (comparable to n_head in simple_train.py)
    
    print("="*60)
    print("Shadow Grouping Method Example")
    print("="*60)
    print(f"Molecule: {mol_name}")
    print(f"Min shots per term: {min_nshot_a_term}")
    print(f"Max shots: {max_nshot}")
    print("="*60)
    
    # Load the Hamiltonian (same as simple_train.py)
    print("\nLoading Hamiltonian...")
    hamil = get_test_hamil("mol", mol_name)
    print(f"Hamiltonian loaded with {len(hamil.terms)} terms")
    print(f"Number of qubits: {hamil.n_qubit}")
    
    # Create Shadow Grouping measurement scheme
    print("\nBuilding Shadow Grouping measurement scheme...")
    print("This uses a derandomized classical shadow approach")
    print("-"*60)
    
    # Initialize shadow grouping
    shadow_grouping = ShadowGroupingMeasurement(hamil, use_weight=True)
    
    # Build the measurement scheme
    measurements = shadow_grouping.build(
        min_nshot_a_term=min_nshot_a_term,
        max_nshot=max_nshot
    )
    
    print("\n" + "="*60)
    print("Shadow Grouping completed!")
    print(f"Number of measurement schemes generated: {len(measurements)}")
    print(f"Each measurement is a Pauli string of length: {len(measurements[0]) if measurements else 0}")
    print("="*60)
    
    # Display some statistics
    print("\nStatistics:")
    print(f"  Total measurement schemes: {len(measurements)}")
    print(f"  Number of Hamiltonian terms: {len(hamil.terms)}")
    print(f"  Measurement efficiency: {len(measurements) / len(hamil.terms):.2f}x")
    
    # Show first few measurement schemes
    print("\nFirst 5 measurement schemes (encoded as Pauli indices):")
    print("  (0=I, 1=X, 2=Y, 3=Z)")
    for i, measurement in enumerate(measurements[:5]):
        # Convert to integers for cleaner display
        measurement_int = measurement.astype(int)
        # Show only non-identity qubits for clarity
        non_identity = np.where(measurement_int != 0)[0]
        print(f"  {i+1}. Measurement {i}: {len(non_identity)} non-identity Paulis")
        if len(non_identity) <= 10:  # Only show details if not too many
            print(f"     Qubits: {non_identity.tolist()}")
            print(f"     Paulis: {measurement_int[non_identity].tolist()}")
    
    print("\nShadow Grouping method finished successfully!")
    
    return measurements, shadow_grouping

if __name__ == '__main__':
    measurements, shadow_grouping = main()

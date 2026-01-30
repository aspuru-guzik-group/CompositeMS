#!/usr/bin/env python
"""
Test loading the generated Methanol Hamiltonian

This script demonstrates how to load and use the Methanol Hamiltonian
after it has been generated.
"""

from composite_ms.hamil import get_test_hamil
import os


def test_load_methanol():
    """Test loading Methanol Hamiltonians."""
    print("="*60)
    print("Testing Methanol Hamiltonian Loading")
    print("="*60)
    
    # Try to find generated methanol files
    hamil_path = os.path.join(
        os.path.dirname(__file__),
        "composite_ms", "hamil", "mol"
    )
    
    print(f"\nLooking for Methanol files in: {hamil_path}")
    
    # List methanol files
    if os.path.exists(hamil_path):
        methanol_files = [f for f in os.listdir(hamil_path) 
                         if f.startswith("CH3OH_") and f.endswith(".op")]
        
        if methanol_files:
            print(f"\nFound {len(methanol_files)} Methanol Hamiltonian(s):")
            for f in sorted(methanol_files):
                print(f"  - {f}")
            
            print("\n" + "-"*60)
            # Load and display info for each
            for filename in sorted(methanol_files):
                name = filename.replace(".op", "")
                try:
                    print(f"\nLoading: {name}")
                    hamil = get_test_hamil("mol", name)
                    print(f"  ✓ Successfully loaded")
                    print(f"  Number of terms: {len(hamil.terms)}")
                    print(f"  Number of qubits: {hamil.n_qubit}")
                    
                    # Show a few sample terms
                    print(f"  Sample terms:")
                    for i, (term, coeff) in enumerate(list(hamil.terms.items())[:3]):
                        if term:  # Skip identity
                            term_str = " ".join([f"{op}{idx}" for idx, op in term])
                            print(f"    {term_str}: {coeff:.6f}")
                        if i >= 2:
                            break
                    
                except Exception as e:
                    print(f"  ✗ Failed to load: {e}")
        else:
            print("\n✗ No Methanol Hamiltonians found.")
            print("Please run 'python generate_methanol.py' first.")
    else:
        print("\n✗ Hamiltonian directory not found.")
    
    print("\n" + "="*60)


def example_usage():
    """Show example usage in a training context."""
    print("\n" + "="*60)
    print("Example: Using Methanol in Training")
    print("="*60)
    
    print("\nExample code:")
    print("```python")
    print("from composite_ms.hamil import get_test_hamil")
    print("from composite_ms.cms_lbcs import train_cms_lbcs, CMS_LBCS_args")
    print("")
    print("# Load Methanol Hamiltonian")
    print("hamil = get_test_hamil('mol', 'CH3OH_38_BK')  # Example")
    print("")
    print("# Set up training")
    print("args = CMS_LBCS_args()")
    print("args.n_step = 100000")
    print("args.lr = 0.01")
    print("")
    print("# Train")
    print("head_ratios, heads = train_cms_lbcs(")
    print("    n_head=20000,")
    print("    hamil=hamil,")
    print("    batch_size=1000,")
    print("    args=args")
    print(")")
    print("```")


if __name__ == '__main__':
    test_load_methanol()
    example_usage()

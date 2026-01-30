"""
Simple training script for CMS-LBCS
This demonstrates how to train the Composite Measurement Scheme with LBCS optimization
"""

from composite_ms.cms_lbcs import train_cms_lbcs, CMS_LBCS_args
from composite_ms.hamil import get_test_hamil

def main():
    # Configuration
    mol_name = "CH3OH_52_JW"  # Using CO2 molecule
    n_head = 20000  # Number of measurement schemes (reduced for faster training)
    batch_size = 1000  # Batch size for training
    
    print("="*60)
    print("Simple CMS-LBCS Training Example")
    print("="*60)
    print(f"Molecule: {mol_name}")
    print(f"Number of heads: {n_head}")
    print(f"Batch size: {batch_size}")
    print("="*60)
    
    # Load the Hamiltonian
    print("\nLoading Hamiltonian...")
    hamil = get_test_hamil("mol", mol_name)
    print(f"Hamiltonian loaded with {len(hamil.terms)} terms")
    print(f"Number of qubits: {hamil.n_qubit}")
    
    # Set up training arguments
    args = CMS_LBCS_args()
    args.n_step = 100000  # Reduced steps for quick demo
    args.lr = 0.01  # Learning rate
    args.verbose = True
    
    print("\nStarting training...")
    print("This will train the measurement scheme to minimize variance")
    print("-"*60)
    
    # Train the model
    head_ratios, heads = train_cms_lbcs(
        n_head=n_head,
        hamil=hamil,
        batch_size=batch_size,
        args=args
    )
    
    print("\n" + "="*60)
    print("Training completed!")
    print(f"Final head_ratios shape: {head_ratios.shape}")
    print(f"Final heads shape: {heads.shape}")
    print("="*60)
    
    # Display some results
    print("\nTop 5 head ratios (measurement scheme weights):")
    import numpy as np
    top_indices = np.argsort(head_ratios)[-5:][::-1]
    for i, idx in enumerate(top_indices):
        print(f"  {i+1}. Head {idx}: {head_ratios[idx]:.6f}")
    
    print("\nTraining finished successfully!")

if __name__ == '__main__':
    main()

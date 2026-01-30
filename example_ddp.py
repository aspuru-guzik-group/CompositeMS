#!/usr/bin/env python
"""
Example demonstrating single-GPU vs multi-GPU training with CMS-LBCS.

For single-GPU:
    python example_ddp.py

For multi-GPU (using DDP):
    torchrun --nproc_per_node=2 example_ddp.py
"""

import torch
import os
from composite_ms.hamil import get_test_hamil
from composite_ms.cms_lbcs import train_cms_lbcs, CMS_LBCS_args


def main():
    # Check if running in distributed mode
    rank = int(os.environ.get('RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    if rank == 0:
        print(f"Running with {world_size} process(es)")
        if torch.cuda.is_available():
            print(f"CUDA GPUs available: {torch.cuda.device_count()}")
    
    # Load Hamiltonian
    mol_name = "H2O_26_JW"
    n_head = 1000
    batch_size = 400
    
    if rank == 0:
        print(f"\nLoading Hamiltonian: {mol_name}")
    
    hamil, _ = get_test_hamil("mol", mol_name).remove_constant()
    
    # Configure training arguments
    # multi_GPU=True enables DistributedDataParallel when launched with torchrun
    args = CMS_LBCS_args(multi_GPU=(world_size > 1))
    args.n_step = 10000  # Reduce steps for quick demo
    args.verbose = True
    
    if rank == 0:
        print(f"\nStarting training with {n_head} heads...")
        print(f"Multi-GPU mode: {args.multi_GPU}")
    
    # Train
    head_ratios, heads = train_cms_lbcs(n_head, hamil, batch_size, args)
    
    # Print results (only from rank 0)
    if rank == 0:
        print("\n" + "="*50)
        print("Training completed!")
        print(f"Head ratios shape: {head_ratios.shape}")
        print(f"Heads shape: {heads.shape}")
        print("="*50)


if __name__ == '__main__':
    main()

#!/usr/bin/env python
"""
Helper script to run CMS-LBCS training with DistributedDataParallel (DDP).

Usage:
    python run_ddp_train.py --n_gpus 2 --n_head 1000 --mol_name H2O_26_JW
    
Or directly with torchrun:
    torchrun --nproc_per_node=2 run_ddp_train.py --mol_name H2O_26_JW
"""

import argparse
import subprocess
import sys
import torch


def main():
    parser = argparse.ArgumentParser(description='Run CMS-LBCS training with DDP')
    parser.add_argument('--n_gpus', type=int, default=None,
                        help='Number of GPUs to use (default: all available)')
    parser.add_argument('--n_head', type=int, default=1000,
                        help='Number of heads')
    parser.add_argument('--mol_name', type=str, default='H2O_26_JW',
                        help='Molecule name')
    parser.add_argument('--batch_size', type=int, default=400,
                        help='Batch size')
    parser.add_argument('--use_torchrun', action='store_true',
                        help='Launch via this script (sets up environment)')
    
    args = parser.parse_args()
    
    # Check if already running in DDP mode
    if 'RANK' in sys.environ or 'LOCAL_RANK' in sys.environ:
        # Already launched by torchrun, execute training
        run_training(args)
    else:
        # Launch with torchrun
        n_gpus = args.n_gpus
        if n_gpus is None:
            n_gpus = torch.cuda.device_count()
            if n_gpus == 0:
                print("No GPUs available. Running on CPU.")
                run_training(args)
                return
        
        print(f"Launching training with {n_gpus} GPUs using torchrun...")
        cmd = [
            'torchrun',
            f'--nproc_per_node={n_gpus}',
            __file__,
            '--use_torchrun',
            '--n_head', str(args.n_head),
            '--mol_name', args.mol_name,
            '--batch_size', str(args.batch_size),
        ]
        
        subprocess.run(cmd)


def run_training(args):
    """Execute the actual training."""
    from composite_ms.hamil import get_test_hamil
    from composite_ms.cms_lbcs import train_cms_lbcs, CMS_LBCS_args
    
    mol_name = args.mol_name
    n_head = args.n_head
    batch_size = args.batch_size
    
    hamil, _ = get_test_hamil("mol", mol_name).remove_constant()
    
    # Enable multi-GPU training
    training_args = CMS_LBCS_args(multi_GPU=True)
    
    head_ratios, heads = train_cms_lbcs(n_head, hamil, batch_size, training_args)
    
    # Only print results from rank 0
    import os
    rank = int(os.environ.get('RANK', 0))
    if rank == 0:
        print("\nTraining completed!")
        print(f"Head ratios shape: {head_ratios.shape}")
        print(f"Heads shape: {heads.shape}")


if __name__ == '__main__':
    main()

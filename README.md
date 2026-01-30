# Install
```bash
git clone https://github.com/aspuru-guzik-group/CompositeMS.git
cd CompositeMS
pip install -r requirements.txt
```

## Documentation

- `SETUP_INSTRUCTIONS.md` - Installation and basic usage
- `HAMILTONIAN_GUIDE.md` - Complete guide to loading and generating Hamiltonians
- `MOLECULES_REFERENCE.md` - Quick reference for available molecules and qubit counts
- `GENERATING_32_QUBITS.md` - How to generate Hamiltonians with 32+ qubits ⭐
- `MULTI_GPU_GUIDE.md` - Multi-GPU training setup

## Quick Start

### Basic Training Example
```bash
python simple_train.py
```

### Generating Custom Hamiltonians (Optional)
To generate new Hamiltonians, install additional dependencies:
```bash
pip install openfermion openfermionpyscf pyscf

# Generate example molecules
python example_generate_hamil.py

# Generate large molecules with 32+ qubits (recommended)
python generate_large_molecules.py

# Or generate Methanol specifically with 6-31g basis
python generate_methanol.py
```

# Multi-GPU Training

CMS-LBCS now supports multi-GPU training using PyTorch's DistributedDataParallel (DDP) for improved performance and scalability.

## Quick Start

### Using the helper script (recommended):
```bash
# Train with all available GPUs
python run_ddp_train.py --n_head 1000 --mol_name H2O_26_JW

# Train with specific number of GPUs
python run_ddp_train.py --n_gpus 2 --n_head 1000 --mol_name H2O_26_JW --batch_size 400
```

### Using torchrun directly:
```bash
# Train with 2 GPUs
torchrun --nproc_per_node=2 composite_ms/cms_lbcs.py

# Train with 4 GPUs
torchrun --nproc_per_node=4 composite_ms/cms_lbcs.py
```

### Programmatic usage:
```python
from composite_ms.hamil import get_test_hamil
from composite_ms.cms_lbcs import train_cms_lbcs, CMS_LBCS_args

hamil, _ = get_test_hamil("mol", "H2O_26_JW").remove_constant()

# Enable multi-GPU training
args = CMS_LBCS_args(multi_GPU=True)
head_ratios, heads = train_cms_lbcs(n_head=1000, hamil=hamil, batch_size=400, args=args)
```

**Note:** When using `multi_GPU=True`, you must launch your script with `torchrun` or `torch.distributed.launch`.

# Benchmark

See the `benchmark` directory
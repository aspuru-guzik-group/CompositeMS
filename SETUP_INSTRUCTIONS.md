# Setup Instructions for CMS-LBCS Training

## Quick Start

### 1. Install Dependencies

First, ensure you have Python 3.7+ installed.

#### Option A: Using uv (Recommended - Much Faster!)

If you have `uv` installed, set up the environment with:

```bash
# Create a virtual environment
uv venv

# Activate the virtual environment
source .venv/bin/activate  # On macOS/Linux
# .venv\Scripts\activate   # On Windows

# Install dependencies
uv pip install -r requirements.txt
```

#### Option B: Using pip

```bash
pip3 install -r requirements.txt
```

Or install packages individually:

```bash
pip3 install sympy numpy torch torchvision torchaudio tqdm matplotlib
```

### 2. Run Simple Training

Once dependencies are installed, activate the virtual environment (if using uv) and run the simple training script:

```bash
# If using uv, first activate the virtual environment
source .venv/bin/activate  # On macOS/Linux
# .venv\Scripts\activate   # On Windows

# Run the training script
python3 simple_train.py
```

This will:
- Load the CO2_30_BK Hamiltonian from the local repository
- Train a measurement scheme with 20000 heads
- Optimize to minimize measurement variance
- Display training progress and results

### 3. Expected Output

You should see:
- Hamiltonian loading information
- Training progress bar with loss/variance values
- Final results with head ratios and shapes

### 4. Customization

You can modify `simple_train.py` to:
- Change the molecule (e.g., "H2O_26_JW")
- Adjust number of heads (n_head)
- Modify batch size
- Change training steps (args.n_step)
- Adjust learning rate (args.lr)

## What is CMS-LBCS?

CMS-LBCS (Composite Measurement Scheme with Large-Batch Coordinate-wise Sampling) is an optimization method for quantum measurement schemes. It minimizes the variance of expectation value estimation by learning optimal measurement bases and their sampling weights.

## Troubleshooting

### CUDA/GPU Issues
If you encounter CUDA errors and want to force CPU usage, add this to the script:
```python
import torch
torch.cuda.is_available = lambda: False
```

### Hamiltonian Files
All pre-generated Hamiltonian files are included in the repository under `composite_ms/hamil/`. Available molecules and lattice models can be found in `composite_ms/hamil/mol/` and `composite_ms/hamil/lattice/`.

### Generating New Hamiltonians (Optional)
If you want to generate new Hamiltonians (not required for basic usage), install additional dependencies:

```bash
pip install openfermion openfermionpyscf pyscf
```

Then you can generate custom Hamiltonians:

```python
from composite_ms.hamil_generator import make_molecular_hamil, get_H2O_geo, save_hamil

# Generate H2O Hamiltonian
geometry = get_H2O_geo(bond_len=0.96)
qubit_hamil, fermion_hamil, energies = make_molecular_hamil(geometry)

# Save it
save_hamil(qubit_hamil, qubit_hamil.n_qubits, "mol", "my_H2O_custom")
```

### Out of Memory
Reduce `n_head` or `batch_size` if you encounter memory issues.

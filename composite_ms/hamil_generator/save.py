"""
Save Hamiltonian to File

Functions to save generated Hamiltonians in the standard format.
"""

import pickle
import os
from pathlib import Path


def save_hamil(hamil, n_qubit, category, title, other_info=None, output_dir=None):
    """
    Save a Hamiltonian to a pickle file.
    
    Args:
        hamil: Hamiltonian operator with .terms attribute (dict)
        n_qubit: Number of qubits
        category: Category subdirectory ("mol" or "lattice")
        title: Filename (without .op extension)
        other_info: Optional dict with additional metadata to save
        output_dir: Optional custom output directory (default: composite_ms/hamil/)
    
    The file will be saved to: {output_dir}/{category}/{title}.op
    """
    data = {
        "term": hamil.terms,
        "n_site": n_qubit
    }
    data.update(other_info or {})
    
    # Determine output directory
    if output_dir is None:
        # Default to composite_ms/hamil/
        this_file_path = os.path.dirname(os.path.abspath(__file__))
        composite_ms_path = os.path.dirname(this_file_path)
        output_dir = os.path.join(composite_ms_path, "hamil")
    
    # Create directory if it doesn't exist
    output_path = Path(output_dir) / category
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save the file
    output_file = output_path / f"{title}.op"
    with open(output_file, "wb") as f:
        pickle.dump(data, f)
    
    print(f"Hamiltonian saved to: {output_file}")
    return str(output_file)

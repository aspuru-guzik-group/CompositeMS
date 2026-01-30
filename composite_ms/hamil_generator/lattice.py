"""
Lattice Hamiltonian Generators

Functions to generate Hamiltonians for lattice models.
"""

try:
    from openfermion import QubitOperator
except ImportError:
    # Fall back to composite_ms QubitOperator if openfermion not available
    from composite_ms.qubit_operator import QubitOperator


def two_dimensional_transverse_field_ising_model(n_col, n_row, g, J=1.0):
    """
    Generate a 2D Transverse Field Ising Model Hamiltonian.
    
    Using the notation of https://en.wikipedia.org/wiki/Transverse-field_Ising_model
    
    Args:
        n_col: Number of columns in the 2D lattice
        n_row: Number of rows in the 2D lattice
        g: Transverse field strength
        J: Coupling constant (default: 1.0)
    
    Returns:
        QubitOperator: The TFIM Hamiltonian
    """
    n_qubit = n_col * n_row

    terms = {}
    # Horizontal nearest-neighbor interactions
    for i in range(n_row):
        for j in range(n_col - 1):
            left = i * n_col + j
            terms[((left, "Z"), (left + 1, "Z"))] = -J
    
    # Vertical nearest-neighbor interactions
    for i in range(n_row - 1):
        for j in range(n_col):
            up = i * n_col + j
            down = up + n_col
            terms[((up, "Z"), (down, "Z"))] = -J

    # Transverse field
    for i in range(n_qubit):
        terms[((i, "X"),)] = -J*g

    op = QubitOperator()
    op.terms = terms
    op.compress()

    return op

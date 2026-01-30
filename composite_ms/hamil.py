from composite_ms.qubit_operator import QubitOperator
import os

this_folder_path = os.path.dirname(os.path.abspath(__file__))
Local_Hamil_Repo_Root = this_folder_path + "/hamil"


def get_test_hamil(category, name, with_constant=False):
    """
    Load a Hamiltonian from the local repository.
    
    Args:
        category: "mol" or "lattice"
        name: Name of the Hamiltonian file (without .op extension)
        with_constant: If False, remove the constant term of the Hamiltonian
    
    Returns:
        QubitOperator: The loaded Hamiltonian
    """
    folder_path = this_folder_path + f"/hamil/{category}/"
    
    try:
        op = QubitOperator.read_op_file(name, folder_path)
    except FileNotFoundError:
        raise Exception(f"Hamiltonian file not found at {folder_path}{name}.op")
    
    # Convert coefficients to real floats
    for pword, coeff in op:
        op.terms[pword] = float(coeff.real)
        assert abs(coeff.imag) < 1e-7
    
    # Remove constant term if requested
    if not with_constant and () in op.terms:
        del op.terms[()]
    
    op.compress()
    return op


if __name__ == '__main__':
    op = get_test_hamil("mol", "H2O_26_BK")
    print(len(op.terms))

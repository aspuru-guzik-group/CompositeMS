from qubit_operator import QubitOperator
from pathlib import Path
import os

this_folder_path = os.path.dirname(os.path.abspath(__file__))
URL_FORM = "https://github.com/doomspec/HamilFactory/blob/main/hamils/{}/{}.op?raw=true"
Local_Hamil_Repo_Root = this_folder_path + "/hamil"


def download_hamil(category, name):
    import wget
    try:
        folder_path = Path("{}/{}/{}".format(Local_Hamil_Repo_Root, category, name)).parent
        folder_path.mkdir(parents=True, exist_ok=True)
        print(name)
        wget.download(URL_FORM.format(category, name), "{}/{}/{}.op".format(Local_Hamil_Repo_Root, category, name))
    except Exception as e:
        print("Failed to Retrieve Hamiltonian from remote.\nError msg: " + str(e))
        return False
    return True


def get_test_hamil(category, name, with_constant=False, try_download=True):
    """
    Please see the files in mizore/testing/hamil/
    Args:
        try_download: If True, try to download from remote if the Hamiltonian is missing
        with_constant: If False, remove the constant of the Hamiltonian
        category: "mol"
    """
    folder_path = this_folder_path + f"/hamil/{category}/"
    try:
        op = QubitOperator.read_op_file(name, folder_path)
    except FileNotFoundError:
        if try_download:
            if download_hamil(category, name):
                op = QubitOperator.read_op_file(name, folder_path)
            else:
                raise Exception("Hamiltonian file not found. Download failed")
        else:
            raise Exception("Hamiltonian file not found")
    for pword, coeff in op:
        op.terms[pword] = float(coeff.real)
        assert abs(coeff.imag) < 1e-7
    if not with_constant and () in op.terms:
        del op.terms[()]
    op.compress()
    return op


if __name__ == '__main__':
    op = get_test_hamil("mol", "H2O_26_BK")
    print(len(op.terms))

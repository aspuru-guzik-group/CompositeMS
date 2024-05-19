from tqdm import trange

from utils import get_ogm_n_heads, project_root
from composite_ms.average_var import average_var_coeff_by_list_of_pwords
from composite_ms.hamil import get_test_hamil
from composite_ms.qubit_operator import QubitOperator
import numpy as np

def f(coeff1, coeff2, pauli_tensor, hamil):

    f1 = average_var_coeff_by_list_of_pwords(
                hamil, QubitOperator.from_pauli_tensor(coeff1, pauli_tensor))
    f2 = average_var_coeff_by_list_of_pwords(
                hamil, QubitOperator.from_pauli_tensor(coeff2, pauli_tensor))
    ratio = np.random.random()
    f_mid = average_var_coeff_by_list_of_pwords(
                hamil, QubitOperator.from_pauli_tensor(ratio*coeff1+(1-ratio)*coeff2, pauli_tensor))
    return ratio * f1 + (1-ratio) * f2, f_mid


def main_normalized():
    hamil_name = "LiH_12_JW"
    get_ogm_n_heads(hamil_name)
    pword_dist = QubitOperator.read_op_file(
            hamil_name, project_root + "/scheme_saved/OG_not_optimized")

    coeff, pauli_tensor = pword_dist.get_pauli_tensor()
    hamil = get_test_hamil("mol", hamil_name)
    coeff1 = np.random.random((len(coeff),))
    coeff2 = np.random.random((len(coeff),))
    coeff1 = coeff1 / sum(coeff1)
    coeff2 = coeff2 / sum(coeff2)
    res = f(coeff1, coeff2, pauli_tensor,hamil)
    assert res[0] > res[1]

def main_not_normalized():
    hamil_name = "LiH_12_JW"
    get_ogm_n_heads(hamil_name)
    pword_dist = QubitOperator.read_op_file(
            hamil_name, project_root + "/scheme_saved/OG_not_optimized")

    coeff, pauli_tensor = pword_dist.get_pauli_tensor()
    hamil = get_test_hamil("mol", hamil_name)
    coeff1 = np.random.random((len(coeff),))
    coeff2 = np.random.random((len(coeff),))
    res = f(coeff1, coeff2, pauli_tensor,hamil)
    assert res[0] > res[1]

if __name__ == '__main__':
    for i in trange(100):
        main_normalized()
    for i in trange(100):
        main_not_normalized()
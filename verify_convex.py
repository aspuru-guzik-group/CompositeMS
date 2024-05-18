from composite_ms.average_var import average_var_coeff_by_list_of_pwords
from composite_ms.hamil import get_test_hamil
from composite_ms.qubit_operator import QubitOperator
from composite_ms.utils import project_root
import numpy as np

def f(coeff1, coeff2, pauli_tensor, hamil):
    coeff1 = coeff1 / sum(coeff1)
    coeff2 = coeff2 / sum(coeff2)
    f1 = average_var_coeff_by_list_of_pwords(
                hamil, QubitOperator.from_pauli_tensor(coeff1, pauli_tensor))
    f2 = average_var_coeff_by_list_of_pwords(
                hamil, QubitOperator.from_pauli_tensor(coeff2, pauli_tensor))
    ratio = np.random.random()
    f_mid = average_var_coeff_by_list_of_pwords(
                hamil, QubitOperator.from_pauli_tensor(ratio*coeff1+(1-ratio)*coeff2, pauli_tensor))
    return ratio * f1 + (1-ratio) * f2, f_mid


def main():
    hamil_name = "LiH_12_JW"
    pword_dist = QubitOperator.read_op_file(
            hamil_name, project_root + "/scheme_saved/OG_not_optimized")

    coeff, pauli_tensor = pword_dist.get_pauli_tensor()
    hamil = get_test_hamil("mol", hamil_name)
    res = f(np.random.random((len(coeff),)),np.random.random((len(coeff),)),pauli_tensor,hamil)
    print(res)
    assert res[0] > res[1]

for i in range(10000):
    main()
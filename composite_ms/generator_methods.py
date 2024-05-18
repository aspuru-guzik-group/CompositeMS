from composite_ms.other_methods.derand import DerandomizationMeasurement
from composite_ms.other_methods.shadow_grouping import ShadowGroupingMeasurement
import numpy as np
from composite_ms.average_var import average_var_coeff_by_list_of_pwords
from composite_ms.qubit_operator import QubitOperator


def get_uncovered_hamil_part(hamil, pword_dist):
    term_coeff, term_tensor = hamil.get_pauli_tensor()
    _, measurement_tensor = pword_dist.get_pauli_tensor()
    cover_count = np.zeros((len(term_tensor),))

    nonzero_mask = (term_tensor != 0)
    for pword_to_measure in measurement_tensor:
        pword_to_measure = np.array(pword_to_measure)
        diff = nonzero_mask * (term_tensor != pword_to_measure)
        is_qwc = np.logical_not(diff.any(axis=-1))
        cover_count += is_qwc

    remaining_tensor = term_tensor[cover_count == 0]
    remaining_coeff = term_coeff[cover_count == 0]
    return QubitOperator.from_pauli_tensor(remaining_coeff, remaining_tensor)

def optimal_mixing_coeff(var1, var2):
    a1 = 1 / (1 + np.sqrt(var2 / var1))
    a2 = 1 - a1
    return a1, a2

def run_generator_ensuring_covering(method_name, n_shot, hamil):
    if method_name == "Derand":
        generator_class = DerandomizationMeasurement
    elif method_name == "ShadowGrouping":
        generator_class = ShadowGroupingMeasurement
    else:
        raise Exception()
    generator = generator_class(hamil)
    pword_tensor = generator.build(-1, max_nshot=n_shot)
    pword_dist = QubitOperator.from_pauli_tensor(np.ones(len(pword_tensor))/len(pword_tensor), pword_tensor)
    remaining_hamil = get_uncovered_hamil_part(hamil, pword_dist)
    measured_hamil = hamil-remaining_hamil
    print(f"{len(measured_hamil)} (out of {len(hamil)}) terms in the Hamiltonian are measured")
    average_var_1 = average_var_coeff_by_list_of_pwords(measured_hamil, pword_dist)
    #print(average_var_1)

    if len(remaining_hamil) == 0:
        return pword_dist

    generator = generator_class(remaining_hamil)
    n_shot //= 2
    pword_tensor_for_remaining = generator.build(-1, max_nshot=n_shot)
    pword_dist_remining = QubitOperator.from_pauli_tensor(np.ones(len(pword_tensor_for_remaining))/len(pword_tensor_for_remaining), pword_tensor_for_remaining)
    average_var_2 = average_var_coeff_by_list_of_pwords(remaining_hamil, pword_dist_remining, allow_uncovered=True)

    super_small_hamil = get_uncovered_hamil_part(remaining_hamil, pword_dist_remining)
    if len(super_small_hamil) != 0:
        print("There are still", len(super_small_hamil), "terms remain. L1 sampling method will be used.")
        c, t = super_small_hamil.get_pauli_tensor()
        c = np.abs(c)
        l1_var = np.sum(c) ** 2
        c = c / np.sum(c)
        l1_dist = QubitOperator.from_pauli_tensor(c, t)
        a1, a2 = optimal_mixing_coeff(average_var_2, l1_var)
        pword_dist_remining = a1 * pword_dist_remining + a2 * l1_dist
        average_var_2 = average_var_coeff_by_list_of_pwords(remaining_hamil, pword_dist_remining)
        print("New var for the remaining part", average_var_2)

    a1, a2 = optimal_mixing_coeff(average_var_1, average_var_2)
    final_dist = pword_dist * a1 + pword_dist_remining * a2
    final_dist /= final_dist.get_l1_norm_omit_const()

    return final_dist
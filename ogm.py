
from itertools import chain
from mizore.operators import QubitOperator
import numpy as np
from tqdm import trange

def get_OGM_grouping(hamil: QubitOperator):

    coeffs, terms = hamil.get_pauli_tensor()
    coeffs = np.abs(coeffs)
    order = np.argsort(-coeffs)
    terms = terms[order, :]
    coeffs = coeffs[order]

    groups = []
    initial_probs = []
    is_term_added = np.array([False] * len(coeffs))
    for i in trange(len(terms), ncols=100):
        if is_term_added[i]:
            continue
        is_term_added[i] = True
        covering_pword = terms[i]
        initial_prob = coeffs[i]

        zeros_in_covering = covering_pword == 0
        for j in chain(range(i + 1, len(terms)), range(i)):
            curr_term = terms[j]
            if ((covering_pword == curr_term) | zeros_in_covering | (curr_term == 0)).all():
                non_zero = (curr_term != 0)
                covering_pword[non_zero] = curr_term[non_zero]
                zeros_in_covering = covering_pword == 0
                is_term_added[j] = True
                initial_prob += coeffs[j]
        groups.append(covering_pword)
        initial_probs.append(initial_prob)

    initial_probs = np.array(initial_probs)
    initial_probs /= np.sum(initial_probs)
    groups = np.array(groups)

    return QubitOperator.from_pauli_tensor(initial_probs, groups)


if __name__ == '__main__':
    from mizore.testing.hamil import get_test_hamil
    from average_var import average_var_coeff_by_list_of_pwords
    hamil = get_test_hamil("mol", "LiH_12_JW")
    op = get_OGM_grouping(hamil)
    print(average_var_coeff_by_list_of_pwords(hamil, op))
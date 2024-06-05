import numpy as np
from numpy import ndarray


def sample_pauli_strings(heads: ndarray, head_ratios: ndarray):
    head_index =  np.random.choice(len(head_ratios), p=head_ratios)
    head = heads[head_index]
    pauli_at_qubits = []
    for i_qubit in range(len(head)):
        pauli_at_qubits.append(np.random.choice(3, p=head[i_qubit]))
    return pauli_at_qubits



if __name__ == '__main__':
    from composite_ms.cms_lbcs import train_cms_lbcs, CMS_LBCS_args
    from composite_ms.hamil import get_test_hamil
    hamil = get_test_hamil("mol", "LiH_12_JW")
    args = CMS_LBCS_args()
    #args.n_step = 100000 # for quick test # delete this line for real use
    heads_ratio, heads = train_cms_lbcs(500, hamil, 500, args)
    # get 100 samples
    for i in range(100):
        pauli_string = sample_pauli_strings(heads, heads_ratio)
        print(pauli_string)

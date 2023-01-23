import numpy as np
import torch
from cms_lbcs import get_no_zero_pauliwords, loss

def average_var_coeff_by_list_of_pwords(ob, pwords_op):
    childer_coeffs, children_tensor = ob.get_pauli_tensor()
    children_cover_count = np.zeros((len(children_tensor),))

    pword_coeffs, pwords_tensor = pwords_op.get_pauli_tensor()

    nonzero_mask = (children_tensor != 0)
    for pword_to_measure, n_shot in zip(pwords_tensor, pword_coeffs):
        pword_to_measure = np.array(pword_to_measure)
        diff = nonzero_mask * (children_tensor != pword_to_measure)
        is_qwc = np.logical_not(diff.any(axis=-1))
        children_cover_count += is_qwc * n_shot
    not_covered_weight = 0.0
    var = 0.0
    for i in range(len(children_tensor)):
        if children_cover_count[i] != 0:
            var += (childer_coeffs[i] ** 2) * (1 / children_cover_count[i])
        else:
            not_covered_weight += abs(childer_coeffs[i])

    if not_covered_weight != 0:
        weight_sum = ob.get_l1_norm_omit_const() 
        print("not_covered_weight:", not_covered_weight)
        print("original var coeff:", var)
        additional_shots = (not_covered_weight / weight_sum) * len(pwords_tensor)
        print("Add", additional_shots, "additional shots")
        var = var + (not_covered_weight**2 / additional_shots) 
        var = var * (additional_shots + len(pwords_tensor)) / len(pwords_tensor)
        print("fixed var coeff:", var)
        raise Exception("Some children is not covered.")
    
    var *= (1 - 1 / (2 ** ob.n_qubit + 1))

    return var

def average_var_coeff_by_cms_lbcs(ratios, heads, ob):
    coeffs, ob_tensor = ob.get_one_hot_tensor()
    coeffs = torch.tensor(coeffs)
    ob_tensor = torch.tensor(ob_tensor)
    no_zero_pauli_tensor = get_no_zero_pauliwords(ob_tensor)
    heads = torch.tensor(heads)
    ratios = torch.tensor(ratios)

    offset = 1 - torch.sum(heads, -1)
    offset = torch.unsqueeze(offset, -1)
    offset = torch.concatenate([offset, torch.zeros((len(heads), len(heads[0]), 2))], -1)
    #offset = torch.concatenate([offset/3, offset/3, offset/3], -1)
    heads += offset

    var = loss(heads, ratios, no_zero_pauli_tensor, coeffs)
    var *= (1 - 1 / (2 ** ob.n_qubit + 1))
    
    return var
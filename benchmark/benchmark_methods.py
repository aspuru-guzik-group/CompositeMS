
from collections import defaultdict
import json
from time import time
import torch
import pickle
from pathlib import Path
from composite_ms.hamil import get_test_hamil

from composite_ms.average_var import average_var_coeff_by_list_of_pwords, average_var_coeff_by_cms_lbcs
from composite_ms.qubit_operator import QubitOperator
import numpy as np
from composite_ms.generator_methods import get_uncovered_hamil_part, optimal_mixing_coeff, run_generator_ensuring_covering
from composite_ms.other_methods.ogm import get_OGM_grouping
from composite_ms.cms_lbcs import train_cms_lbcs, CMS_LBCS_args
from utils import project_root, result_root, create_if_not_exist


def run_part_of_benchmark(hamil_name, method_name, read_file=False):

    hamil = get_test_hamil("mol", hamil_name)
    hamil_name = hamil_name.replace("/", "-")

    method_folder = project_root+"/scheme_saved/"+method_name
    res_in_cms_lbcs = None

    if read_file:
        if method_name in ["De", "SG"]:
            pword_dist = QubitOperator.read_op_file(hamil_name, method_folder)
            # var_coeff = average_var_coeff_by_list_of_pwords(hamil, pword_dist)
            res_in_cms_lbcs = pword_dist.get_one_hot_tensor()
            remaining_hamil = get_uncovered_hamil_part(hamil, pword_dist)
            truncated_weight = remaining_hamil.get_l1_norm_omit_const()
        else:
            with open(method_folder+"/"+hamil_name+".pkl", "rb") as f:
                res_in_cms_lbcs = pickle.load(f)
        if truncated_weight != 0.0:
            truncated_hamil = hamil - remaining_hamil
            print("Compensate for truncation", truncated_weight)
            var_coeff = average_var_coeff_by_list_of_pwords(
                truncated_hamil, pword_dist)
            print("Var before", var_coeff)
            a1, a2 = optimal_mixing_coeff(var_coeff, truncated_weight ** 2)
            var_coeff = var_coeff / a1 + (truncated_weight ** 2) / a2
            print("Var after", var_coeff)
        else:
            var_coeff = float(average_var_coeff_by_cms_lbcs(
                *res_in_cms_lbcs, hamil))
        return var_coeff

    pword_dist = None
    if method_name == "SG":
        c, t = hamil.get_pauli_tensor()
        r = np.abs(c / np.max(c))
        not_too_small = r > 1e-7
        truncated_weight = np.sum(np.abs(c[np.logical_not(not_too_small)]))
        print("Truncated weight:", truncated_weight)
        c = c[not_too_small]
        t = t[not_too_small]
        hamil = QubitOperator.from_pauli_tensor(c, t)
        pword_dist = run_generator_ensuring_covering(
            "ShadowGrouping", len(hamil) * 3, hamil)

    if method_name == "De":
        pword_dist = run_generator_ensuring_covering(
            "Derand", len(hamil) * 3, hamil)
    if method_name == "OG":
        try:
            init_pword_dist = QubitOperator.read_op_file(
                hamil_name, project_root + "/scheme_saved/OG_not_optimized")
            print("Un-optimized OGM grouping loaded from existing file.")
        except:
            init_pword_dist = get_OGM_grouping(hamil)
        # Save the un-optimized OGM groups
        init_pword_dist.save_to_op_file(
            hamil_name, project_root + "/scheme_saved/OG_not_optimized")
        args = CMS_LBCS_args()
        head_ratios, heads = init_pword_dist.get_one_hot_tensor()
        heads = torch.asarray(heads)
        offset = 1 - torch.sum(heads, -1)
        offset = torch.unsqueeze(offset, -1)
        offset = torch.concatenate(
            [offset, torch.zeros((len(heads), len(heads[0]), 2))], -1)
        heads += offset
        args.set_init_heads(head_ratios, heads * 100)
        args.freeze_heads = True
        batch_size = 500  # len(hamil)
        res_in_cms_lbcs = train_cms_lbcs(-1, hamil, batch_size, args)

    if method_name == "CMS":
        time_start = time()
        ogm_pword_dist = QubitOperator.read_op_file(
            hamil_name, project_root + "/scheme_saved/OG_not_optimized")
        n_heads = len(ogm_pword_dist)
        res_in_cms_lbcs = train_cms_lbcs(n_heads, hamil, 500, CMS_LBCS_args())
        time_used = time()-time_start
        log_time_used(hamil_name, method_name, time_used)

    if method_name == "CMS2":
        ogm_pword_dist = QubitOperator.read_op_file(
            hamil_name, project_root + "/scheme_saved/OG_not_optimized")
        n_heads = len(ogm_pword_dist) * 2
        res_in_cms_lbcs = train_cms_lbcs(n_heads, hamil, 1000, CMS_LBCS_args())

    if pword_dist is not None:
        var_coeff = average_var_coeff_by_list_of_pwords(hamil, pword_dist)
        print(var_coeff)

    if res_in_cms_lbcs is not None:
        Path(method_folder).mkdir(parents=True, exist_ok=True)
        with open(method_folder+"/"+hamil_name+".pkl", "wb") as f:
            pickle.dump(res_in_cms_lbcs, f)
    else:
        pword_dist.save_to_op_file(hamil_name, method_folder)
        res_in_cms_lbcs = pword_dist.get_one_hot_tensor()

    var_coeff = float(average_var_coeff_by_cms_lbcs(*res_in_cms_lbcs, hamil))

    if method_name == "SG":
        if truncated_weight > 1e-8:
            a1, a2 = optimal_mixing_coeff(var_coeff, truncated_weight ** 2)
            var_coeff = var_coeff / a1 + (truncated_weight ** 2) / a2

    print("var_coeff for", method_name, ":", var_coeff)

    return var_coeff


def run_benchmark(mol_list, method_list):
    create_if_not_exist(result_root + "/var_vs_methods.json")
    for method in method_list:
        for mol in mol_list:
            for trans in ["BK"]:
                mol_name = mol + "_" + trans
                print("Working on", mol_name)
                var_coeff = run_part_of_benchmark(
                    mol_name, method, read_file=False)
                with open(result_root + "/var_vs_methods.json", "r") as f:
                    res_dict = defaultdict(lambda: {})
                    res_dict.update(json.load(f))
                res_dict[mol_name][method] = var_coeff
                with open(result_root + "/var_vs_methods.json", "w") as f:
                    json.dump(dict(res_dict), f, indent=1)

def log_time_used(mol_name, method, time_used):
    create_if_not_exist(result_root + "/time_used.json")
    with open(result_root + "/time_used.json", "r") as f:
        time_dict = defaultdict(lambda: {})
        time_dict.update(json.load(f))
    time_dict[mol_name][method] = time_used
    with open(result_root + "/time_used.json", "w") as f:
        json.dump(dict(time_dict), f, indent=1)

full_mol_list = ["LiH_12", "H_chain/H6_12", "H2O_14",
                 "NH3_16", "N2_20", "C2H2_24", "C2H4_28", "CO2_30"]
method_list = ["OG", "SG", "De", "CMS"]


def run():
    run_benchmark(full_mol_list[:], method_list[3:4])
    #run_benchmark(["CO2_30"], method_list[3:4])


if __name__ == '__main__':
    run()

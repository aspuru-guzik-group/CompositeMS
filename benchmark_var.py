
from collections import defaultdict
import json, torch
from mizore.testing.hamil import get_test_hamil

from average_var import average_var_coeff_by_list_of_pwords, average_var_coeff_by_cms_lbcs
from mizore.operators import QubitOperator
import numpy as np
from generator_methods import run_generator_ensuring_covering
from ogm import get_OGM_grouping
from cms_lbcs import train_cms_lbcs, CMS_LBCS_args
from utils import project_root



def run_part_of_benchmark(hamil_name, method_name):

    hamil = get_test_hamil("mol", hamil_name)
    res_in_cms_lbcs = None
    pword_dist = None
    if method_name == "SG":
        pword_dist = run_generator_ensuring_covering("ShadowGrouping", 3000, hamil)
    if method_name == "De":
        pword_dist = run_generator_ensuring_covering("Derand", 3000, hamil)
    if method_name == "OG":
        try:
            init_pword_dist = QubitOperator.read_op_file(hamil_name, project_root+ "/ogm_groups")
            print("Un-optimized OGM grouping loaded from existing file.")
        except:
            init_pword_dist = get_OGM_grouping(hamil)
        # Save the un-optimized OGM groups
        init_pword_dist.save_to_op_file(hamil_name, project_root + "/ogm_groups")
        args = CMS_LBCS_args()
        head_ratios, heads = init_pword_dist.get_one_hot_tensor()
        heads = torch.asarray(heads)
        offset = 1 - torch.sum(heads, -1)
        offset = torch.unsqueeze(offset, -1)
        offset = torch.concatenate([offset, torch.zeros((len(heads), len(heads[0]), 2))], -1)
        heads += offset
        args.set_init_heads(head_ratios, heads * 100)
        args.freeze_heads = True
        batch_size = 500 # len(hamil)
        res_in_cms_lbcs = train_cms_lbcs(-1, hamil, batch_size, args)

    if method_name == "CMS":
        ogm_pword_dist = QubitOperator.read_op_file(hamil_name, project_root+ "/ogm_groups")
        n_heads = len(ogm_pword_dist)
        res_in_cms_lbcs = train_cms_lbcs(n_heads, hamil, 500, CMS_LBCS_args())

    if pword_dist is not None:
        var_coeff = average_var_coeff_by_list_of_pwords(hamil, pword_dist)
        print(var_coeff)

    if not res_in_cms_lbcs:
        res_in_cms_lbcs = pword_dist.get_one_hot_tensor()

    var_coeff = float(average_var_coeff_by_cms_lbcs(*res_in_cms_lbcs, hamil))

    print("var_coeff for", method_name, ":",var_coeff)

    return var_coeff


def run_benchmark(mol_list, method_list):
    with open(project_root + "/var_coeff_for_many.json", "r") as f:
        res_dict = defaultdict(lambda: {})
        res_dict.update(json.load(f))

    for method in method_list:
        for mol in mol_list:
            for trans in ["JW", "BK"]:
                mol_name = mol + "_" + trans
                print("Working on", mol_name)
                var_coeff = run_part_of_benchmark(mol_name, method)
                res_dict[mol_name][method] = var_coeff
                with open(project_root + "/var_coeff_for_many.json", "w") as f:
                    json.dump(dict(res_dict), f, indent=1)

full_mol_list = ["LiH_12", "H_chain/H6_12", "H2O_14", "NH3_16", "N2_20", "C2H2_24", "C2H4_28", "CO2_30"]
method_list = ["OG", "SG", "De", "CMS"]
def run():
    run_benchmark(full_mol_list, method_list[1:])


if __name__ == '__main__':
    run()
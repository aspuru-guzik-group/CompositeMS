from collections import defaultdict
import json
import math
from qubit_operator import QubitOperator
from hamil import get_test_hamil
from average_var import average_var_coeff_by_cms_lbcs
from ogm import get_OGM_grouping
from utils import project_root
from cms_lbcs import train_cms_lbcs, CMS_LBCS_args


mol_list = ["N2_20_JW", "C2H2_24_JW", "C2H4_28_JW", "CO2_30_JW"]
#mol_list = ["H_chain/H6_12_JW", "H_chain/H8_16_JW", "H_chain/H10_20_JW", "H_chain/H12_24_JW"]
for hamil_name in mol_list[:]:
    hamil = get_test_hamil("mol", hamil_name)
    hamil_name = hamil_name.replace("/","-")
    try:
        OGM_n_head = len(QubitOperator.read_op_file(
            hamil_name, project_root + "/scheme_saved/OG_not_optimized"))
    except:
        init_pword_dist = get_OGM_grouping(hamil)
        # Save the un-optimized OGM groups
        init_pword_dist.save_to_op_file(hamil_name, project_root + "/scheme_saved/OG_not_optimized")
        OGM_n_head = len(init_pword_dist)

    n_head = OGM_n_head
    settings = ["bi-level rescale", "bi-level", "rescale", "nothing"]
    for turned_on in settings[0:1]:
        args = CMS_LBCS_args()
        turned_on_list = turned_on.split(" ")
        if "bi-level" not in turned_on:
            args.bilevel_ratio = 1.0
            print("bi-level turned off")
        if "rescale" not in turned_on:
            args.rescale_head_grad = False
            print("rescale turned off")
        if "more-bi-level" in turned_on:
            args.bilevel_ratio = 0.05
            args.n_non_decreasing_step_to_stop *= 2
            print("more bi-level turned on")

        if "bi-level" in turned_on:
            pass
            #args.ratio_adam_beta_1 = 0.8
            #args.bilevel_ratio = 0.05
            #args.alternate_training_ratio = -0.5
            #args.alternate_training_n_steps = -2
            #args.n_non_decreasing_step_to_stop *= 2
            #print(args.alternate_training_ratio, args.alternate_training_n_steps)

        #res_in_cms_lbcs = train_cms_lbcs(n_head, hamil, len(hamil) // 80 + 1, args)
        res_in_cms_lbcs = train_cms_lbcs(n_head, hamil, 500, args)
        var_coeff = average_var_coeff_by_cms_lbcs(*res_in_cms_lbcs, hamil)

        with open(project_root + "/var_vs_optimization.json", "r") as f:
            res_dict = defaultdict(lambda: {})
            res_dict.update(json.load(f))
            var_coeffs = res_dict[hamil_name]
        var_coeffs[turned_on] = float(var_coeff)
        with open(project_root + "/var_vs_optimization.json", "w") as f:
            json.dump(dict(res_dict), f, indent=2)

from collections import defaultdict
import json
from composite_ms.hamil import get_test_hamil
from composite_ms.average_var import average_var_coeff_by_cms_lbcs
from utils import result_root, get_ogm_n_heads, create_if_not_exist
from composite_ms.cms_lbcs import train_cms_lbcs, CMS_LBCS_args


def main(hamil_name, settings):
    result_path = result_root + "/var_vs_optimization.json"
    create_if_not_exist(result_path)
    hamil = get_test_hamil("mol", hamil_name)
    hamil_name = hamil_name.replace("/","-")
    OGM_n_head = get_ogm_n_heads(hamil_name)

    n_head = OGM_n_head
    for turned_on in settings:
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

        with open(result_path, "r") as f:
            res_dict = defaultdict(lambda: {})
            res_dict.update(json.load(f))
            var_coeffs = res_dict[hamil_name]
        var_coeffs[turned_on] = float(var_coeff)
        with open(result_path, "w") as f:
            json.dump(dict(res_dict), f, indent=2)


if __name__ == '__main__':
    mol_list = ["LiH_12_JW","N2_20_JW", "C2H2_24_JW", "C2H4_28_JW", "CO2_30_JW"]
    settings = ["bi-level rescale", "bi-level", "rescale", "nothing"]
    # mol_list = ["H_chain/H6_12_JW", "H_chain/H8_16_JW", "H_chain/H10_20_JW", "H_chain/H12_24_JW"]
    for hamil_name in mol_list[0:1]:
        main(hamil_name, settings[0:1])
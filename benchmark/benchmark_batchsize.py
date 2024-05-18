from collections import defaultdict
import json
import math
from composite_ms.hamil import get_test_hamil
from composite_ms.average_var import average_var_coeff_by_cms_lbcs
from composite_ms.cms_lbcs import train_cms_lbcs, CMS_LBCS_args

from utils import result_root, get_ogm_n_heads, create_if_not_exist


def main(hamil_name):
    create_if_not_exist(result_root + "/var_vs_batchsize.json")
    hamil_name = hamil_name + "_" + "BK"
    OGM_n_head = get_ogm_n_heads(hamil_name)
    hamil = get_test_hamil("mol", hamil_name)
    batch_sizes_divider = [256, 128, 64, 32]
    for i in batch_sizes_divider[:1]:
        n_head = OGM_n_head
        batch_size = math.ceil(len(hamil) / i)
        print("batch_size", batch_size, "for", hamil_name)
        args = CMS_LBCS_args()
        args.n_non_decreasing_step_to_stop = args.n_non_decreasing_step_to_stop * (500 / batch_size)
        res_in_cms_lbcs = train_cms_lbcs(n_head, hamil, batch_size, args)
        with open(result_root + "/var_vs_batchsize.json", "r") as f:
            res_dict = defaultdict(lambda: {})
            res_dict.update(json.load(f))
            var_coeffs = res_dict[hamil_name]
        var_coeff = average_var_coeff_by_cms_lbcs(*res_in_cms_lbcs, hamil)
        var_coeffs[batch_size] = float(var_coeff)
        with open(result_root + "/var_vs_batchsize.json", "w") as f:
            json.dump(dict(res_dict), f, indent=2)


if __name__ == '__main__':
    mol_list = ["H2O_14", "NH3_16", "N2_20", "C2H2_24", "C2H4_28", "CO2_30"]
    for hamil_name in mol_list[:4]:
        main(hamil_name)
from collections import defaultdict
import json
import math

from composite_ms.hamil import get_test_hamil
from composite_ms.average_var import average_var_coeff_by_cms_lbcs
from utils import result_root, get_ogm_n_heads, create_if_not_exist
from composite_ms.cms_lbcs import train_cms_lbcs, CMS_LBCS_args



def main(hamil_name):
    create_if_not_exist(result_root + "/var_vs_n_heads.json")
    OGM_n_head = get_ogm_n_heads(hamil_name)
    hamil = get_test_hamil("mol", hamil_name)
    args = CMS_LBCS_args()
    for i in [8, 4, 2, 1, 0.5]:
        n_head = math.ceil(OGM_n_head / i / 10) * 10
        print("n_head", n_head, "for", hamil_name)
        res_in_cms_lbcs = train_cms_lbcs(n_head, hamil, 500, args)
        with open(result_root + "/var_vs_n_heads.json", "r") as f:
            res_dict = defaultdict(lambda: {})
            res_dict.update(json.load(f))
            var_coeffs = res_dict[hamil_name]
        var_coeff = average_var_coeff_by_cms_lbcs(*res_in_cms_lbcs, hamil)
        var_coeffs[n_head] = float(var_coeff)
        with open(result_root + "/var_vs_n_heads.json", "w") as f:
            json.dump(dict(res_dict), f, indent=2)

if __name__ == '__main__':
    mol_list = ["H2O_14_JW", "NH3_16_JW", "N2_20_JW", "C2H2_24_JW", "C2H4_28_JW",
                "CO2_30_JW"]
    for hamil_name in mol_list[0:1]:
        main(hamil_name)
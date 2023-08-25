from collections import defaultdict
import json
import math
from qubit_operator import QubitOperator
from hamil import get_test_hamil
from average_var import average_var_coeff_by_cms_lbcs
from utils import project_root
from cms_lbcs import train_cms_lbcs, CMS_LBCS_args


mol_list = ["H2O_14_JW", "NH3_16_JW", "N2_20_JW", "C2H2_24_JW", "C2H4_28_JW", "CO2_30_JW"]
for hamil_name in mol_list[-2:]:
    OGM_n_head = len(QubitOperator.read_op_file(
        hamil_name, project_root + "/scheme_saved/OG_not_optimized"))
    hamil = get_test_hamil("mol", hamil_name)
    args = CMS_LBCS_args()
    for i in [8, 4, 2, 1, 0.5]:
        n_head = math.ceil(OGM_n_head / i / 10) * 10
        print("n_head", n_head, "for", hamil_name)
        res_in_cms_lbcs = train_cms_lbcs(n_head, hamil, 500, args)
        with open(project_root + "/var_vs_n_heads.json", "r") as f:
            res_dict = defaultdict(lambda: {})
            res_dict.update(json.load(f))
            var_coeffs = res_dict[hamil_name]
        var_coeff = average_var_coeff_by_cms_lbcs(*res_in_cms_lbcs, hamil)
        var_coeffs[n_head] = float(var_coeff)
        with open(project_root + "/var_vs_n_heads.json", "w") as f:
            json.dump(dict(res_dict), f, indent=2)

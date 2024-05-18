import os

from composite_ms.hamil import get_test_hamil
from composite_ms.other_methods.ogm import get_OGM_grouping
from composite_ms.qubit_operator import QubitOperator

project_root = os.path.dirname(os.path.abspath(__file__))
result_root = project_root + "/results"

def get_ogm_n_heads(hamil_name):
    try:
        return len(QubitOperator.read_op_file(
            hamil_name, project_root + "/scheme_saved/OG_not_optimized"))
    except FileNotFoundError:
        hamil = get_test_hamil("mol", hamil_name)
        init_pword_dist = get_OGM_grouping(hamil)
        # Save the un-optimized OGM groups
        init_pword_dist.save_to_op_file(hamil_name,
                                        project_root + "/scheme_saved/OG_not_optimized")
        OGM_n_head = len(init_pword_dist)
        return OGM_n_head

def create_if_not_exist(path):
    if not os.path.exists(path):
        with open(path, "w") as f:
            f.write("{}")

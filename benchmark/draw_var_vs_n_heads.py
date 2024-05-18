import json
import math
import os
from utils import result_root, get_ogm_n_heads
import matplotlib.pyplot as plt
import numpy as np


def to_tex_mol_name(hamil_name):
    res = []
    hamil_name: str
    hamil_name = hamil_name.split("/")[-1]
    in_brace = False
    for s in hamil_name:
        if s.isnumeric():
            if not in_brace:
                res.append("_{")
                in_brace = True
        else:
            if in_brace:
                res.append("}")
                in_brace = False
        res.append(s)
    if in_brace:
        res.append("}")
    return "$\\mathrm{"+"".join(res)+"}$"

# %%
with open(os.path.dirname(os.path.abspath(__file__)) + "/var_vs_n_heads.json", "r") as f:
    res_dict = json.load(f)
with open(os.path.dirname(os.path.abspath(__file__)) + "/var_vs_methods.json", "r") as f:
    other_method_dict = json.load(f)

hamil_names = list(res_dict.keys())  # ["LiH_12_JW", "H2O_14_JW", "NH3_16_JW", "N2_20_JW"]
print(hamil_names)

n_hamils = len(hamil_names)
ncols = 3
nrows = math.ceil(n_hamils / 3)
fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(3.5 * ncols, 2.5 * nrows))

for i in range(n_hamils):
    hamil_name = hamil_names[i]
    hamil_dict = res_dict[hamil_name]
    heads = np.array([int(head) for head in hamil_dict.keys()])
    vars = np.array([var for var in hamil_dict.values()])
    ranks = np.argsort(heads)
    heads = heads[ranks]
    vars = vars[ranks]
    heads = heads[1:]
    vars = vars[1:]
    curr_ax = ax[i // 3][i % 3]
    latex_mol_name = to_tex_mol_name("_".join(hamil_name.split("_")[:-2]))

    curr_ax.set_title(latex_mol_name)
    # heads = -1/heads
    try:
        OGM_n_head = get_ogm_n_heads(hamil_name)
        ogm_var = other_method_dict[hamil_name]["SG"]
        curr_ax.axhline(y=ogm_var, color="#b85d3e", linestyle='--', label="SG", linewidth=3)
        curr_ax.axvline(x=OGM_n_head, color='#dfba42', linestyle='--', linewidth=3)
    except:
        pass

    curr_ax.plot(heads, vars, "-o", color="#497954", label="C-LBCS", linewidth=3)
   
    curr_ax.set_xlabel("Number of sub-schemes")
    curr_ax.set_ylabel("Average one-shot variance")
    # curr_ax.set_xscale('log')
    curr_ax.legend()

plt.subplots_adjust(hspace=0.1)
plt.subplots_adjust(wspace=0.05)
plt.tight_layout()
plt.show()
fig.savefig(result_root + f"/n_scheme_compare.pdf")

# %%

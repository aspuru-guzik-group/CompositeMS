import json
import math
from utils import project_root


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

with open(project_root + "/var_vs_methods.json", "r") as f:
    res_dict = json.load(f)


method_name_map = {
    "De": "Derand.", 
    "OG": "OGM", 
    "SG": "SG",
    "CMS": "C-LBCS"
}

hamil_names = ["LiH_12", "H_chain/H6_12", "H2O_14", "NH3_16", "N2_20", "C2H2_24", "C2H4_28", "CO2_30"]
methods = ["De", "OG", "SG", "CMS"]


res = []
res.append("\\begin{table}[!h]\\centering\n\\begin{tabular}")
res.append("{c|c|"+"c"*(len(methods))+"}")
res.append("\n\\toprule \n")
res.append("Molecule & Enc. ")
res.append("".join([f" & {method_name_map[method]}" for method in methods])+"\\\\ \n\hline \n")

transforms = ["JW", "BK"]
n_trans = len(transforms)
for hamil in hamil_names:
    hamil_info = hamil.split("/")[-1].split("_")
    print(hamil_info)
    tex_hamil_name = to_tex_mol_name(hamil_info[0])
    res.extend(["\\multirow{",str(n_trans),"}{*}{",f"{tex_hamil_name}({hamil_info[1]})","}","\n"])
    for i_trans in range(len(transforms)):
        transform = transforms[i_trans]
        res.append(f"& {transform} ")
        hamil_dict = res_dict.get(hamil+"_"+transform, {})
        var_coeff_list = []
        for method in methods:
            if method in hamil_dict:
                method_dict = hamil_dict[method]
            else:
                print(method, "not in", hamil_dict)
                method_dict = {}
            var_coeff = method_dict
            if var_coeff < 100:
                var_coeff = "{:0.2f}".format(var_coeff)
            else:
                var_coeff = round(var_coeff)
                var_coeff = "{}".format(var_coeff)
            var_coeff_list.append(var_coeff)
        
        min_coeff = min([float(x) for x in var_coeff_list])
        for var_coeff in var_coeff_list:
            if float(var_coeff) == min_coeff:
                res.extend(["& \\textbf{", f"{var_coeff}","}"])
            else:
                res.append(f"& {var_coeff}")
        res.append("\\\\")
    res.append("\n\\hline \n")


res.append("""\\botrule\\end{tabular} 
\\vartablecaption
\\end{table}
""")

print("".join(res))
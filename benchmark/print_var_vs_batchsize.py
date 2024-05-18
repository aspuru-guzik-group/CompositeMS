import json
import math

from composite_ms.hamil import get_test_hamil
from utils import result_root


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

with open(result_root + "/var_vs_batchsize.json", "r") as f:
    res_dict = json.load(f)


def method_name_map(times):
    return "$1/"+str(times)+"$"

hamil_names = ["H2O_14", "NH3_16", "N2_20", "C2H2_24", "C2H4_28", "CO2_30"]
methods = [128, 64, 32]


res = []
res.append("\\begin{table}[!h]\\centering\n\\begin{tabular}")
res.append("{c|c|c"+"c"*(len(methods))+"}")
res.append("\n\\toprule \n")
res.append("Molecule & $n_H$")
res.append("".join([f" & {method_name_map(method)}" for method in methods])+"\\\\ \n\hline \n")

transforms = ["BK"]
n_trans = len(transforms)
for hamil in hamil_names:
    hamil_info = hamil.split("/")[-1].split("_")
    tex_hamil_name = to_tex_mol_name(hamil_info[0])
    res.extend(["\\multirow{",str(n_trans),"}{*}{",f"{tex_hamil_name}({hamil_info[1]})","}","\n"])
    for i_trans in range(len(transforms)):
        transform = transforms[i_trans]
        hamil_op = get_test_hamil("mol", hamil + "_" + transform)
        res.append(f"& {len(hamil_op)} ")
        hamil = hamil.replace("/", "-")
        hamil_dict = res_dict.get(hamil+"_"+transform, {})
        print(hamil_dict)
        var_coeff_list = []
        for method in methods:
            key = str(math.ceil(len(hamil_op) / method))
            if key in hamil_dict:
                var_coeff = hamil_dict[key]
            else:
                print(hamil)
                print(key, "not in", hamil_dict)
                var_coeff = -1
            var_coeff = "{:0.2f}".format(var_coeff)
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
\\vartablecaptionBatch
\\end{table}
""")

print("".join(res))
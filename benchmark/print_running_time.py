import json
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

with open(result_root + "/time_used.json", "r") as f:
    res_dict = json.load(f)


hamil_names = ["LiH_12", "H2O_14", "NH3_16", "N2_20", "C2H2_24", "C2H4_28", "CO2_30"]
methods = ["Time used (s)"]
res = []
res.append("\\begin{table}[!h]\\centering\n\\begin{tabular}")
res.append("{c|c|"+"c"*(len(methods))+"}")
res.append("\n\\toprule \n")
res.append("Molecule & Enc. ")
res.append("".join([f" & {method}" for method in methods])+"\\\\ \n\hline \n")

transforms = ["JW", "BK"]
n_trans = len(transforms)
for hamil in hamil_names:
    hamil_key = hamil.replace("/", "-")
    hamil_info = hamil.split("/")[-1].split("_")
    #print(hamil_info)
    tex_hamil_name = to_tex_mol_name(hamil_info[0])
    res.extend(["\\multirow{",str(n_trans),"}{*}{",f"{tex_hamil_name}({hamil_info[1]})","}","\n"])
    #hamil = get_test_hamil("mol", hamil_key + "_JW")
    #print(len(hamil.terms))
    for i_trans in range(len(transforms)):
        transform = transforms[i_trans]
        res.append(f"& {transform} ")
        hamil_dict = res_dict.get(hamil_key+"_"+transform, {})
        var_coeff_list = []
        for method in methods:
            time_used = hamil_dict["CMS"]
            var_coeff_list.append(time_used)
            if time_used < 100:
                time_used = "{:0.1f}".format(time_used)
            else:
                time_used = round(time_used)
                time_used = "{}".format(time_used)
            res.append(f"& {time_used}")
            if transform == "BK":
                print(time_used)
        res.append("\\\\")
    res.append("\n\\hline \n")


res.append("""\\botrule\\end{tabular} 
\\timetablecaption
\\end{table}
""")

print("".join(res))
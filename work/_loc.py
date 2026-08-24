import os

p = r"D:\codefiles\python\diordered-circuits\model-dev\learning\src\learning\experiments\run_vafidis_toy.py"
txt = open(p, encoding="utf-8").read()
out = []
i = txt.find("def run_fixed_point_map_diagnostic")
out.append("def at line %d" % (txt[:i].count(chr(10)) + 1))
j = txt.find('if "fixed_point_map" in enabled:')
out.append("run_all_tests branch at line %d" % (txt[:j].count(chr(10)) + 1))
out.append("---branch---")
out.append(txt[j - 150 : j + 600])
with open(r"D:\codefiles\python\diordered-circuits\work\_loc.txt", "w", encoding="utf-8") as fh:
    fh.write("\n".join(out))
print("done")

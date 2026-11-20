# merge the per-seed image files of shapes3d_factors.py into one, taking each prior from its best checkpoint.
#   python analysis/shapes3d_best_images.py <dir with s<seed>_images.npz> <out prefix>
import sys, numpy as np
d, out = sys.argv[1], sys.argv[2]
PICK = {"C": 4, "N": 2, "S": 2, "N_L2": 1}
z = {s: np.load(f"{d}/s{s}_images.npz") for s in set(PICK.values())}
m = {}
for k in z[4].files:
    parts = k.split("|")
    if parts[0] in ("true", "src"): m[k] = z[4][k]; continue
    p = parts[1] if parts[0] == "ok" else parts[0]
    m[k] = z[PICK[p]][k]
np.savez_compressed(out + "_images.npz", **m)
for p in PICK: print(p, "seed", PICK[p], {f: (int(m[f"ok|{p}|{f}|bind"].sum()), int(m[f"ok|{p}|{f}|vector"].sum()), len(m[f"true|{f}"])) for f in ("orientation", "scale", "floor hue", "wall hue")})

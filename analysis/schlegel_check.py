# unbinding of bundled pairs (schlegel et al. 2022, sec. 4.3 protocol shape): k role-filler pairs, query one role,
# nearest neighbour among 1000 items. four vector families, each at "D dimensions" as schlegel counts them:
#   hrr          D real numbers, N(0, 1/D), unbind with the involution (approximate inverse)
#   fhrr         D complex phasors (D angles = 2D real numbers when written out), unbind with the conjugate (exact)
#   unitary D    a real vector of length D with unit fourier magnitudes (what our reference uses): D/2 free angles
#   unitary 2D   the same at real length 2D: D free angles, i.e. an fhrr vector of D phasors written as a real vector
#   python analysis/schlegel_check.py <out.json>
import numpy as np, json, sys
rng = np.random.default_rng(0)
N = 1000
def hrr(n, D): return rng.normal(0, 1 / np.sqrt(D), (n, D))
def unitary(n, D):
    ph = rng.uniform(-np.pi, np.pi, (n, D // 2 + 1)); ph[:, 0] = 0
    if D % 2 == 0: ph[:, -1] = 0
    return np.fft.irfft(np.exp(1j * ph), n=D, axis=1) * np.sqrt(D) / np.sqrt(D)  # unit fourier magnitudes
def run_real(make, D, k, trials):
    ok = 0
    for _ in range(trials):
        M = make(N, D); F = np.fft.rfft(M, axis=1)
        ri = rng.choice(N, k, replace=False); fi = rng.choice(N, k, replace=False)
        B = (F[ri] * F[fi]).sum(0)
        dec = np.fft.irfft(B[None] * np.conj(F[ri]), n=D, axis=1)
        Mn = M / np.linalg.norm(M, axis=1, keepdims=True)
        ok += ((dec @ Mn.T).argmax(1) == fi).mean()
    return ok / trials
def run_fhrr(D, k, trials, normalize):
    ok = 0
    for _ in range(trials):
        M = np.exp(1j * rng.uniform(-np.pi, np.pi, (N, D)))
        ri = rng.choice(N, k, replace=False); fi = rng.choice(N, k, replace=False)
        B = (M[ri] * M[fi]).sum(0)
        if normalize: B = B / np.abs(B)   # schlegel's fhrr bundling keeps only the angles of the sum
        dec = B[None] * np.conj(M[ri])
        ok += ((dec @ np.conj(M).T).real.argmax(1) == fi).mean()
    return ok / trials
FAM = {"hrr": lambda D, k, t: run_real(hrr, D, k, t), "fhrr (angles of the sum)": lambda D, k, t: run_fhrr(D, k, t, True),
       "fhrr (plain sum)": lambda D, k, t: run_fhrr(D, k, t, False),
       "unitary real, length D": lambda D, k, t: run_real(unitary, D, k, t), "unitary real, length 2D": lambda D, k, t: run_real(unitary, 2 * D, k, t)}
out = {"acc": {}, "mind": {}}
# 1. accuracy at fixed D
for D in (128, 256, 512):
    for k in (5, 10, 15, 20, 30):
        out["acc"][f"{D}|{k}"] = {n: round(f(D, k, 60), 3) for n, f in FAM.items()}
        print(D, k, out["acc"][f"{D}|{k}"], flush=True)
# 2. smallest D reaching 99% (schlegel fig. 8), scanning D upward in steps of 16
for k in (5, 10, 15, 20, 25, 30):
    row = {}
    for n, f in FAM.items():
        D = 16
        while f(D, k, 30) < 0.99: D += 16
        row[n] = D
    out["mind"][k] = row; print("min D for 99%:", k, row, flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)

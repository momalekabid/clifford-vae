import time, torch, math
from power_spherical import PowerSpherical
from scipy.special import i0, i1
from clifford import (
    CliffordTorusDistribution as VM_Torus,
    CliffordPowerSphericalDistribution as PS_Torus,
)

torch.manual_seed(42)
B, D, S = 3, 4096, 4096           # batch, latent‑dim, sample‑size
loc   = torch.zeros(B, D)
kappa = torch.ones (B, D) * 5.0


def check_symmetry():
    x = PS_Torus(loc, kappa).rsample((256,))
    f = torch.fft.fft(x, dim=-1, norm="ortho")  # correct arg order
    ok = torch.allclose(f[..., 1:D],
                        torch.conj(torch.flip(f[..., -D+1:], (-1,))),
                        atol=1e-5)
    print("conjugate‑symmetry   :", "PASS" if ok else "FAIL")

# --- 4. speed ratio ----------------------------------------------------------
def check_speed():
    t0 = time.time(); VM_Torus(loc,kappa).rsample((1024,)); t_vm = time.time()-t0
    t0 = time.time(); PS_Torus(loc,kappa).rsample((1024,)); t_ps = time.time()-t0
    print(f"speed VM/PS          : {t_vm/t_ps:.2f} × ")

if __name__ == "__main__":
    check_symmetry()
    check_speed()

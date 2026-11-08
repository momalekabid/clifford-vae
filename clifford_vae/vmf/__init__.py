# vendored from nicola-decao/s-vae-pytorch (davidson et al. 2018), patched: float64 log-bessel
# (the float32 cast and +1e-20 in the log made the kl negative at d >= 40).
from .von_mises_fisher import VonMisesFisher
from .hyperspherical_uniform import HypersphericalUniform

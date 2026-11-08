# vaes for every prior, one interface.
#   N      gaussian                     N_L2   gaussian, mu and z l2-normalised
#   S      power spherical              V      von mises-fisher (mlp only)
#   C      clifford torus: power spherical on each circle, one kappa per circle, floor 0.01 (cnn)
#          or 0.03 (mlp), cap 10. kappa_shared=True gives the submission's one kappa per image.
# cnn: residual conv encoder/decoder, l1 reconstruction, 32px (4 blocks) or 64px (5 blocks).
# mlp: 784-256-128-z, bernoulli likelihood on binarised mnist (davidson et al. 2018).
# N_L2: the kl is the gaussian kl of the pre-normalisation latent; normalisation is the first
# layer of the decoder. parameter names match the submission code, so old checkpoints load.
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dists import CliffordPowerSpherical, CliffordTorusUniform, HypersphericalUniform, PowerSpherical, clifford_embed

PRIORS = ["N", "N_L2", "S", "V", "C"]
CLIFFORD = ("C",)

# power spherical / vmf: kappa cap 50 x ambient dim (a fixed cap of 10 makes S near-uniform at
# d=128 and pins kappa with zero gradient). clifford keeps a cap of 10: with no cap kappa climbs
# past 20 and role-filler breaks. one kappa per circle with a floor of 0.01 beats one shared kappa
# at d >= 512 on elbo, fid and bundle capacity (3 seeds, cifar-10 and fashionmnist): unused circles
# go flat while the rest stay sharp, whereas the shared kappa sinks to its floor at large d.
KAPPA_CAP_PER_DIM = 50.0
CLIFFORD_KAPPA_CAP = 10.0
CLIFFORD_FLOOR = {"cnn": 0.01, "mlp": 0.03}


def _clifford_floor(d):
    # floor of the shared kappa: grows with d so the circles don't all collapse to uniform at high d
    for lim, f in ((255, 0.04), (512, 0.07), (1024, 0.10), (2048, 0.13)):
        if d <= lim:
            return f
    return 0.16


def _xavier(m):
    if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)


# ---------------------------------------------------------------- cnn


class ResBlock(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.conv = nn.Conv2d(i, o, 4, 2, 1)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.skip = nn.Conv2d(i, o, 1) if i != o else nn.Identity()
        self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        return self.act(self.conv(x)) + self.pool(self.skip(x))


class ResUpBlock(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.conv = nn.ConvTranspose2d(i, o, 4, 2, 1)
        self.act = nn.LeakyReLU(0.2, inplace=True)
        self.skip = nn.Conv2d(i, o, 1) if i != o else nn.Identity()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")

    def forward(self, x):
        return self.act(self.conv(x)) + self.up(self.skip(x))


class Encoder(nn.Module):
    def __init__(self, d, channels, dist, l2, img_size, floor, cap=CLIFFORD_KAPPA_CAP, per_circle=False):
        super().__init__()
        self.dist, self.l2, self.floor, self.cap, self.latent_dim = dist, l2, floor, cap, d
        chs = [channels, 64, 128, 256, 512] + ([512] if img_size == 64 else [])
        self.blocks = nn.ModuleList([ResBlock(chs[i], chs[i + 1]) for i in range(len(chs) - 1)])
        self.fc_mu = nn.Linear(2048, d)
        if dist == "N":
            self.fc_log_var = nn.Linear(2048, d)
        else:
            self.fc_concentration = nn.Linear(2048, d if per_circle else 1)
        self.apply(_xavier)

    def forward(self, x):
        for b in self.blocks:
            x = b(x)
        x = x.flatten(1)
        mu = self.fc_mu(x)
        if self.dist == "N":
            return (F.normalize(mu, dim=-1) if self.l2 else mu), self.fc_log_var(x)
        if self.dist == "S":
            k = F.softplus(self.fc_concentration(x)) + 0.5
            return F.normalize(mu, dim=-1), k.clamp(max=KAPPA_CAP_PER_DIM * self.latent_dim)
        k = F.softplus(self.fc_concentration(x)) + self.floor
        return mu, k.clamp(max=self.cap)


class Decoder(nn.Module):
    def __init__(self, d_in, channels, img_size):
        super().__init__()
        self.fc = nn.Linear(d_in, 2048)
        chs = ([512] if img_size == 64 else []) + [512, 256, 128, 64]
        self.blocks = nn.ModuleList([ResUpBlock(chs[i], chs[i + 1]) for i in range(len(chs) - 1)])
        self.final = nn.Sequential(nn.ConvTranspose2d(chs[-1], channels, 4, 2, 1), nn.Tanh())
        self.apply(_xavier)

    def forward(self, z):
        x = self.fc(z).view(len(z), 512, 2, 2)
        for b in self.blocks:
            x = b(x)
        return self.final(x)


class CNNVAE(nn.Module):
    arch = "cnn"

    def __init__(self, prior, d, channels, img_size, kappa_cap=None, kappa_floor=None, kappa_shared=False):
        super().__init__()
        self.prior, self.latent_dim, self.kappa_shared = prior, d, prior == "C" and kappa_shared
        dist = {"N": "N", "N_L2": "N", "S": "S", "C": "C"}[prior]
        per_circle = prior == "C" and not self.kappa_shared
        default = CLIFFORD_FLOOR["cnn"] if per_circle else _clifford_floor(d)
        self.kappa_floor = (default if kappa_floor is None else kappa_floor) if prior in CLIFFORD else 0.0
        self.kappa_cap = CLIFFORD_KAPPA_CAP if kappa_cap is None else kappa_cap
        self.encoder = Encoder(d, channels, dist, prior == "N_L2", img_size, self.kappa_floor, self.kappa_cap, per_circle)
        self.decoder = Decoder(2 * d if prior in CLIFFORD else d, channels, img_size)

    def posterior(self, mu, p):
        dev = mu.device
        if self.prior in ("N", "N_L2"):
            return torch.distributions.Normal(mu, torch.exp(0.5 * p) + 1e-6), torch.distributions.Normal(0.0, 1.0)
        if self.prior == "S":
            return PowerSpherical(mu, p.squeeze(-1)), HypersphericalUniform(self.latent_dim, dev)
        return CliffordPowerSpherical(mu, p), CliffordTorusUniform(self.latent_dim, dev)

    def sample_code(self, x):
        """one posterior sample per image: the decoder input in training."""
        mu, p = self.encoder(x)
        q, _ = self.posterior(mu, p)
        z = q.rsample()
        return F.normalize(z, dim=-1) if self.prior == "N_L2" else z

    def loss(self, x, beta):
        mu, p = self.encoder(x)
        q, pz = self.posterior(mu, p)
        z = q.rsample()
        if self.prior == "N_L2":
            z = F.normalize(z, dim=-1)
        recon = F.l1_loss(self.decoder(z), x, reduction="sum") / len(x)
        kl = torch.distributions.kl_divergence(q, pz)
        kl = (kl.sum(-1) if kl.dim() > 1 else kl).mean()
        return recon + beta * kl, recon, kl

    def kappa(self, x):
        return self.encoder(x)[1]

    def encode_mean(self, x):
        return self.encoder(x)[0]


# ---------------------------------------------------------------- mlp


class MLPVAE(nn.Module):
    arch = "mlp"

    def __init__(self, prior, d, x_dim=784, kappa_cap=None, kappa_floor=None, kappa_shared=False):
        super().__init__()
        self.prior, self.latent_dim, self.x_dim = prior, d, x_dim
        self.kappa_shared = prior == "C" and kappa_shared
        self.kappa_cap = CLIFFORD_KAPPA_CAP if kappa_cap is None else kappa_cap
        self.kappa_floor = CLIFFORD_FLOOR["mlp"] if kappa_floor is None else kappa_floor
        # S and V live on S^d in R^{d+1} (davidson et al. 2018 convention)
        self.z_dim = d + 1 if prior in ("S", "V") else d
        self.encoder = nn.Sequential(nn.Linear(x_dim, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU())
        self.fc_mean = nn.Linear(128, self.z_dim)
        if prior in ("N", "N_L2"):
            self.fc_var = nn.Linear(128, self.z_dim)
        else:
            self.fc_scale = nn.Linear(128, self.z_dim if prior in CLIFFORD and not self.kappa_shared else 1)
        d_in = 2 * self.z_dim if prior in CLIFFORD else self.z_dim
        self.decoder_net = nn.Sequential(nn.Linear(d_in, 128), nn.ReLU(), nn.Linear(128, 256), nn.ReLU(),
                                         nn.Linear(256, x_dim))
        self.apply(_xavier)

    def decoder(self, z):
        """logits -> images in [0, 1] (same role as the cnn decoder's tanh output)."""
        return torch.sigmoid(self.decoder_net(z))

    def _encode(self, x):
        h = self.encoder(x.view(len(x), -1))
        mu = self.fc_mean(h)
        if self.prior in ("N", "N_L2"):
            return (F.normalize(mu, dim=-1) if self.prior == "N_L2" else mu), self.fc_var(h)
        if self.prior in ("S", "V"):
            k = F.softplus(self.fc_scale(h)) + 0.8
            return F.normalize(mu, dim=-1), k.clamp(max=KAPPA_CAP_PER_DIM * self.z_dim)
        return mu, (F.softplus(self.fc_scale(h)) + self.kappa_floor).clamp(max=self.kappa_cap)

    def posterior(self, mu, p):
        dev = mu.device
        if self.prior in ("N", "N_L2"):
            return torch.distributions.Normal(mu, torch.exp(0.5 * p) + 1e-6), torch.distributions.Normal(0.0, 1.0)
        if self.prior == "S":
            return PowerSpherical(mu, p.squeeze(-1)), HypersphericalUniform(self.z_dim, dev)
        if self.prior == "V":
            from .vmf import HypersphericalUniform as VU, VonMisesFisher

            return VonMisesFisher(mu, p), VU(self.z_dim - 1, device=dev)
        return CliffordPowerSpherical(mu, p), CliffordTorusUniform(self.z_dim, dev)

    def sample_code(self, x):
        mu, p = self._encode(x)
        z = self.posterior(mu, p)[0].rsample()
        return F.normalize(z, dim=-1) if self.prior == "N_L2" else z

    def loss(self, x, beta):
        mu, p = self._encode(x)
        q, pz = self.posterior(mu, p)
        z = q.rsample()
        if self.prior == "N_L2":
            z = F.normalize(z, dim=-1)
        recon = F.binary_cross_entropy_with_logits(self.decoder_net(z), x.view(len(x), -1), reduction="sum") / len(x)
        # kl summed over latent dims (the submission averaged it, i.e. kl/d for N and N_L2)
        kl = torch.distributions.kl_divergence(q, pz)
        kl = (kl.sum(-1) if kl.dim() > 1 else kl).mean()
        return recon + beta * kl, recon, kl

    def kappa(self, x):
        return self._encode(x)[1]

    def encode_mean(self, x):
        return self._encode(x)[0]

    def load_state_dict(self, state, strict=True):
        # submission checkpoints call the decoder "decoder.*"
        state = {k.replace("decoder.", "decoder_net.", 1) if k.startswith("decoder.") else k: v for k, v in state.items()}
        return super().load_state_dict(state, strict)


# ---------------------------------------------------------------- shared


def build(prior, d, data, dev, kappa_cap=None, kappa_floor=None, kappa_shared=False):
    """kappa_cap / kappa_floor / kappa_shared: clifford only, None / False = the defaults (cap 10,
    floor 0.01 cnn / 0.03 mlp, one kappa per circle)."""
    if data.arch == "cnn":
        m = CNNVAE(prior, d, data.channels, data.img_size, kappa_cap, kappa_floor, kappa_shared)
    else:
        m = MLPVAE(prior, d, data.channels * data.img_size * data.img_size, kappa_cap, kappa_floor, kappa_shared)
    return m.to(dev)


def kappa_shared_in(path):
    """whether a clifford checkpoint was trained with one kappa per image (the submission and every
    run before the per-circle default): its kappa head has a single output."""
    state = torch.load(path, map_location="cpu", weights_only=False)["state"]
    w = state.get("encoder.fc_concentration.weight", state.get("fc_scale.weight"))
    return w is not None and w.shape[0] == 1


def mean_code(model, x):
    """the deterministic code every evaluation uses. N: mu. N_L2: mu/|mu|. S, V: mean direction.
    C: the length-2d unitary vector at the mean angles (never the raw angles)."""
    mu = model.encode_mean(x)
    return clifford_embed(mu) if model.prior in CLIFFORD else mu


@torch.no_grad()
def mean_codes(model, images, bs=1000):
    return torch.cat([mean_code(model, images.get(idx)).float() for idx in images.batches(bs)])


@torch.no_grad()
def sampled_codes(model, images, bs=1000, seed=0):
    """one posterior sample per image (what the submission's vsa tests used), seeded. for clifford
    the sampled unitary vector, like the mean code."""
    torch.manual_seed(seed)
    return torch.cat([model.sample_code(images.get(idx)).float() for idx in images.batches(bs)])


def prior_samples(model, n, gen):
    dev = next(model.parameters()).device
    d = model.z_dim if model.arch == "mlp" else model.latent_dim
    if model.prior in CLIFFORD:
        a = torch.rand(n, d, generator=gen) * 2 * torch.pi
        return clifford_embed(a).to(dev)
    z = torch.randn(n, d, generator=gen).to(dev)
    # uniform on the sphere = normalised gaussian
    return F.normalize(z, dim=-1) if model.prior in ("N_L2", "S", "V") else z


def kappa_stats(model, x):
    """mean kappa and the fraction of kappas at the cap (S, V, C) and, for clifford, at the floor
    (softplus ~ 0: the encoder wants the circle flatter than it is allowed to be)."""
    if model.prior in ("N", "N_L2"):
        return {}
    with torch.no_grad():
        k = model.kappa(x)
    if model.prior in CLIFFORD:
        cap = model.kappa_cap
    else:
        cap = KAPPA_CAP_PER_DIM * (model.z_dim if model.arch == "mlp" else model.latent_dim)
    out = {"kappa_mean": k.mean().item(), "kappa_at_cap": (k >= cap * 0.999).float().mean().item()}
    if model.prior in CLIFFORD:
        out["kappa_at_floor"] = (k <= model.kappa_floor + 1e-3).float().mean().item()
    return out


def n_params(model):
    return sum(p.numel() for p in model.parameters())

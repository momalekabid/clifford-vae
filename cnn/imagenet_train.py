# imagenet 256x256 training for CliffordARVAE
# supports clifford, powerspherical, and gaussian distributions
# matches SphereAR S-VAE training setup with perceptual + adversarial losses
# usage: torchrun --nproc_per_node=8 cnn/imagenet_train.py --data-path /path/to/data ...

import argparse
import math
import os
import sys
import time
import json
import numpy as np
from collections import OrderedDict
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
import torchvision
import torchvision.transforms as transforms
import torchvision.utils as tv_utils

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cnn.cliffordar_model import CliffordARVAE
from utils.wandb_utils import WandbLogger


# ---- patchgan discriminator ----


class PatchGANDiscriminator(nn.Module):
    """70x70 patchgan discriminator for adversarial training."""

    def __init__(self, in_channels=3, ndf=64, n_layers=3):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, ndf, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        ch = ndf
        for i in range(1, n_layers):
            ch_prev = ch
            ch = min(ch * 2, 512)
            layers += [
                nn.Conv2d(ch_prev, ch, 4, stride=2, padding=1, bias=False),
                nn.GroupNorm(min(32, ch // 4), ch),
                nn.LeakyReLU(0.2, inplace=True),
            ]
        ch_prev = ch
        ch = min(ch * 2, 512)
        layers += [
            nn.Conv2d(ch_prev, ch, 4, stride=1, padding=1, bias=False),
            nn.GroupNorm(min(32, ch // 4), ch),
            nn.LeakyReLU(0.2, inplace=True),
        ]
        layers += [nn.Conv2d(ch, 1, 4, stride=1, padding=1)]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# ---- lpips perceptual loss using vgg16 ----


class LPIPS(nn.Module):
    """perceptual loss via pretrained vgg16 features.
    extracts features at relu1_2, relu2_2, relu3_3, relu4_3, relu5_3.
    """

    def __init__(self):
        super().__init__()
        vgg = torchvision.models.vgg16(
            weights=torchvision.models.VGG16_Weights.IMAGENET1K_V1
        )
        # feature extraction layers (after relu)
        slices = [4, 9, 16, 23, 30]
        self.blocks = nn.ModuleList()
        prev = 0
        for s in slices:
            self.blocks.append(nn.Sequential(*list(vgg.features.children())[prev:s]))
            prev = s
        for p in self.parameters():
            p.requires_grad = False
        # learned linear weights per layer
        channels = [64, 128, 256, 512, 512]
        self.linears = nn.ModuleList([nn.Conv2d(c, 1, 1, bias=False) for c in channels])
        # init weights to 1/n_layers for reasonable starting point
        for lin in self.linears:
            nn.init.constant_(lin.weight, 1.0 / len(channels))

    def forward(self, x, y):
        # normalize from [-1,1] to vgg input range
        mean = torch.tensor([0.485, 0.456, 0.406], device=x.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=x.device).view(1, 3, 1, 1)
        x = (x * 0.5 + 0.5 - mean) / std
        y = (y * 0.5 + 0.5 - mean) / std

        loss = 0.0
        feat_x, feat_y = x, y
        for block, lin in zip(self.blocks, self.linears):
            feat_x = block(feat_x)
            feat_y = block(feat_y)
            # normalize channel-wise, then compute l2 distance
            diff = (F.normalize(feat_x, dim=1) - F.normalize(feat_y, dim=1)) ** 2
            loss = loss + lin(diff).mean()
        return loss


# ---- tar-based imagenet dataset ----


class ImageNetTarDataset(Dataset):
    """reads imagenet from a single tar file.
    builds an index on first access, then seeks into the tar for each sample.
    """

    def __init__(self, tar_path, transform=None):
        import tarfile

        self.tar_path = tar_path
        self.transform = transform

        # build index: list of (offset, size) for each jpeg
        print(f"indexing tar file: {tar_path} ...")
        self.members = []
        self.class_to_idx = {}
        with tarfile.open(tar_path, "r") as tf:
            for member in tf:
                if not member.isfile():
                    continue
                name = member.name.lower()
                if not (
                    name.endswith(".jpeg")
                    or name.endswith(".jpg")
                    or name.endswith(".png")
                ):
                    continue
                # extract class from path: typically ILSVRC.../nXXXXXXXX/nXXXXXXXX_XXXXX.JPEG
                parts = member.name.split("/")
                cls_name = None
                for p in parts:
                    if p.startswith("n") and len(p) == 9:
                        cls_name = p
                        break
                if cls_name is None:
                    cls_name = parts[-2] if len(parts) > 1 else "unknown"
                if cls_name not in self.class_to_idx:
                    self.class_to_idx[cls_name] = len(self.class_to_idx)
                self.members.append(
                    (member.name, member.offset_data, member.size, cls_name)
                )
        print(f"found {len(self.members)} images in {len(self.class_to_idx)} classes")

    def __len__(self):
        return len(self.members)

    def __getitem__(self, idx):
        from PIL import Image
        import io

        name, offset, size, cls_name = self.members[idx]
        with open(self.tar_path, "rb") as f:
            f.seek(offset)
            data = f.read(size)
        img = Image.open(io.BytesIO(data)).convert("RGB")
        label = self.class_to_idx[cls_name]
        if self.transform:
            img = self.transform(img)
        return img, label


# ---- utilities ----


def setup_distributed():
    """initialize ddp if launched via torchrun."""
    if "RANK" in os.environ:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        return rank, world_size, device
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return 0, 1, device


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


def reduce_tensor(tensor):
    """average tensor across all processes."""
    if not dist.is_initialized():
        return tensor
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt


def save_checkpoint(state, path):
    torch.save(state, path)


def load_checkpoint(path, model, optimizer_vae=None, optimizer_disc=None, scaler=None):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    # handle DDP state dicts
    state_dict = ckpt["model"]
    new_sd = OrderedDict()
    for k, v in state_dict.items():
        new_sd[k.replace("module.", "")] = v
    model.load_state_dict(new_sd)
    if optimizer_vae and "optimizer_vae" in ckpt:
        optimizer_vae.load_state_dict(ckpt["optimizer_vae"])
    if optimizer_disc and "optimizer_disc" in ckpt:
        optimizer_disc.load_state_dict(ckpt["optimizer_disc"])
    if scaler and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    return ckpt.get("epoch", 0), ckpt.get("global_step", 0)


def get_cosine_schedule_with_warmup(
    optimizer, warmup_steps, total_steps, min_lr_ratio=0.0
):
    """cosine lr decay with linear warmup."""

    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (
            1.0 + math.cos(math.pi * progress)
        )

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


@torch.no_grad()
def save_reconstructions(model, images, path, nrow=8):
    """save side-by-side original vs reconstruction."""
    model.eval()
    x_recon, _, _, _ = model(images)
    x_recon = x_recon.clamp(-1, 1)
    # interleave originals and recons
    n = min(images.size(0), nrow)
    comparison = torch.stack([images[:n], x_recon[:n]], dim=1).reshape(
        -1, *images.shape[1:]
    )
    tv_utils.save_image(
        comparison, path, nrow=nrow, normalize=True, value_range=(-1, 1)
    )
    model.train()


@torch.no_grad()
def save_samples_npz(
    model, num_samples, latent_dim, distribution, device, path, batch_size=64
):
    """generate samples and save as npz for FID computation."""
    model.eval()
    all_samples = []
    num_tokens = (
        model.module.num_tokens if hasattr(model, "module") else model.num_tokens
    )
    _model = model.module if hasattr(model, "module") else model

    for i in range(0, num_samples, batch_size):
        bs = min(batch_size, num_samples - i)
        if distribution == "clifford":
            # sample uniform angles, construct conjugate-symmetric spectrum, IFFT to bivector
            angles = torch.rand(bs, num_tokens, latent_dim, device=device) * 2 * math.pi
            n = 2 * latent_dim
            theta_s = torch.zeros(bs, num_tokens, n, device=device)
            theta_s[..., 1:latent_dim] = angles[..., 1:]
            theta_s[..., -latent_dim + 1 :] = -torch.flip(angles[..., 1:], dims=(-1,))
            z = torch.fft.ifft(torch.exp(1j * theta_s), dim=-1).real
        elif distribution == "powerspherical":
            z = torch.randn(bs, num_tokens, latent_dim, device=device)
            z = F.normalize(z, dim=-1) * (latent_dim**0.5)
        else:
            z = torch.randn(bs, num_tokens, latent_dim, device=device)
        imgs = _model.decoder(z).clamp(-1, 1)
        # convert to uint8 [0, 255]
        imgs = ((imgs + 1) * 127.5).clamp(0, 255).to(torch.uint8)
        all_samples.append(imgs.cpu().numpy())

    all_samples = np.concatenate(all_samples, axis=0)
    np.savez(path, arr_0=all_samples)
    model.train()
    return all_samples.shape[0]


# ---- hinge loss for discriminator ----


def hinge_d_loss(logits_real, logits_fake):
    return 0.5 * (F.relu(1.0 - logits_real).mean() + F.relu(1.0 + logits_fake).mean())


def hinge_g_loss(logits_fake):
    return -logits_fake.mean()


# ---- main training ----


def create_dataset(args):
    """create imagenet dataset from tar or folder."""
    transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                args.image_size,
                scale=(0.8, 1.0),
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),  # -> [-1, 1]
        ]
    )
    if args.data_path.endswith(".tar"):
        dataset = ImageNetTarDataset(args.data_path, transform=transform)
    else:
        dataset = torchvision.datasets.ImageFolder(args.data_path, transform=transform)
    return dataset


def train(args):
    rank, world_size, device = setup_distributed()

    if is_main_process():
        os.makedirs(args.results_dir, exist_ok=True)
        os.makedirs(os.path.join(args.results_dir, "checkpoints"), exist_ok=True)
        os.makedirs(os.path.join(args.results_dir, "samples"), exist_ok=True)
        # save args
        with open(os.path.join(args.results_dir, "args.json"), "w") as f:
            json.dump(vars(args), f, indent=2)

    # model
    model = CliffordARVAE(
        latent_dim=args.latent_dim,
        image_size=args.image_size,
        in_channels=3,
        distribution=args.distribution,
        device="cpu",  # move to device after ddp wrapping
        recon_loss_type="l1",
        l1_weight=args.reconstruction_weight,
        patch_size=args.patch_size if args.patch_size > 0 else None,
    ).to(device)

    # discriminator and perceptual loss
    disc = PatchGANDiscriminator(in_channels=3).to(device)
    lpips_loss = LPIPS().to(device)

    if is_main_process():
        n_params = sum(p.numel() for p in model.parameters()) / 1e6
        n_disc = sum(p.numel() for p in disc.parameters()) / 1e6
        print(f"vae params: {n_params:.1f}M, disc params: {n_disc:.1f}M")
        print(f"distribution: {args.distribution}, latent_dim: {args.latent_dim}")

    # ddp wrapping
    if world_size > 1:
        model = DDP(model, device_ids=[device.index], find_unused_parameters=False)
        disc = DDP(disc, device_ids=[device.index], find_unused_parameters=False)

    # optimizers
    optimizer_vae = torch.optim.AdamW(
        list(model.parameters()) + list(lpips_loss.linears.parameters()),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.05,
    )
    optimizer_disc = torch.optim.AdamW(
        disc.parameters(),
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=0.05,
    )

    # dataset
    dataset = create_dataset(args)
    per_gpu_batch = args.global_batch_size // world_size
    sampler = (
        DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
        if world_size > 1
        else None
    )
    loader = DataLoader(
        dataset,
        batch_size=per_gpu_batch,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    steps_per_epoch = len(loader)
    total_steps = args.epochs * steps_per_epoch

    # lr schedulers
    scheduler_vae = get_cosine_schedule_with_warmup(
        optimizer_vae, args.warmup_steps, total_steps
    )
    scheduler_disc = get_cosine_schedule_with_warmup(
        optimizer_disc, args.warmup_steps, total_steps
    )

    # mixed precision
    scaler = torch.amp.GradScaler("cuda", enabled=args.use_amp)
    autocast_ctx = lambda: torch.amp.autocast(
        "cuda", dtype=torch.bfloat16, enabled=args.use_amp
    )

    # wandb
    wlogger = None
    if is_main_process() and not args.no_wandb:
        wlogger = WandbLogger(args)
        run_name = f"imagenet_{args.distribution}_d{args.latent_dim}"
        wlogger.start_run(run_name, args)

    # resume
    start_epoch = 0
    global_step = 0
    if args.resume:
        ckpt_path = args.resume
        if is_main_process():
            print(f"resuming from {ckpt_path}")
        _model = model.module if hasattr(model, "module") else model
        start_epoch, global_step = load_checkpoint(
            ckpt_path, _model, optimizer_vae, optimizer_disc, scaler
        )
        # re-wrap if needed (state dict already loaded)
        if world_size > 1:
            model = DDP(_model, device_ids=[device.index], find_unused_parameters=False)

    # grab a fixed batch for reconstruction vis
    vis_images = None

    # adversarial loss warmup: start discriminator after disc_start_step
    disc_start_step = args.disc_start_step

    if is_main_process():
        print(
            f"training for {args.epochs} epochs, {steps_per_epoch} steps/epoch, {total_steps} total steps"
        )
        print(f"per-gpu batch: {per_gpu_batch}, global batch: {args.global_batch_size}")

    for epoch in range(start_epoch, args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        model.train()
        disc.train()

        epoch_losses = {
            k: 0.0 for k in ["total", "recon", "kld", "perceptual", "gen", "disc"]
        }
        epoch_samples = 0
        t0 = time.time()

        for step, (images, _) in enumerate(loader):
            images = images.to(device, non_blocking=True)

            # grab vis batch once
            if vis_images is None and is_main_process():
                vis_images = images[:16].clone()

            # ---- train generator (vae + lpips) ----
            optimizer_vae.zero_grad(set_to_none=True)

            with autocast_ctx():
                x_recon, q_z, p_z, mu = model(images)
                _model = model.module if hasattr(model, "module") else model
                # compute_loss with beta=1.0 gives total = recon + kld; we weight kld ourselves
                losses = _model.compute_loss(images, x_recon, q_z, p_z, beta=1.0)

                recon_loss = losses["recon_loss"]
                kld_loss = losses["kld_loss"]

                # perceptual loss
                p_loss = lpips_loss(images, x_recon) * args.perceptual_weight

                # adversarial loss (generator side)
                use_disc = (
                    global_step >= disc_start_step and args.adversarial_weight > 0
                )
                if use_disc:
                    logits_fake = disc(x_recon)
                    g_loss = hinge_g_loss(logits_fake) * args.adversarial_weight
                else:
                    g_loss = torch.tensor(0.0, device=device)

                vae_loss = (
                    args.reconstruction_weight * recon_loss
                    + args.kl_weight * kld_loss
                    + p_loss
                    + g_loss
                )

            scaler.scale(vae_loss).backward()
            scaler.unscale_(optimizer_vae)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            torch.nn.utils.clip_grad_norm_(
                lpips_loss.linears.parameters(), args.grad_clip
            )
            scaler.step(optimizer_vae)
            scaler.update()
            scheduler_vae.step()

            # ---- train discriminator ----
            if use_disc:
                optimizer_disc.zero_grad(set_to_none=True)
                with autocast_ctx():
                    logits_real = disc(images.detach())
                    logits_fake = disc(x_recon.detach())
                    d_loss = hinge_d_loss(logits_real, logits_fake)

                scaler.scale(d_loss).backward()
                scaler.unscale_(optimizer_disc)
                torch.nn.utils.clip_grad_norm_(disc.parameters(), args.grad_clip)
                scaler.step(optimizer_disc)
                scaler.update()
                scheduler_disc.step()
            else:
                d_loss = torch.tensor(0.0, device=device)
                scheduler_disc.step()

            # accumulate stats
            bs = images.size(0)
            epoch_losses["total"] += vae_loss.item() * bs
            epoch_losses["recon"] += recon_loss.item() * bs
            epoch_losses["kld"] += kld_loss.item() * bs
            epoch_losses["perceptual"] += p_loss.item() * bs
            epoch_losses["gen"] += g_loss.item() * bs
            epoch_losses["disc"] += d_loss.item() * bs
            epoch_samples += bs
            global_step += 1

            # logging
            if global_step % args.log_every == 0 and is_main_process():
                avg = {k: v / max(epoch_samples, 1) for k, v in epoch_losses.items()}
                lr = optimizer_vae.param_groups[0]["lr"]
                elapsed = time.time() - t0
                samples_per_sec = epoch_samples / max(elapsed, 1e-6) * world_size
                print(
                    f"[epoch {epoch+1}/{args.epochs}] step {step+1}/{steps_per_epoch} "
                    f"| total={avg['total']:.4f} recon={avg['recon']:.4f} kl={avg['kld']:.4f} "
                    f"perc={avg['perceptual']:.4f} g={avg['gen']:.4f} d={avg['disc']:.4f} "
                    f"| lr={lr:.2e} | {samples_per_sec:.0f} img/s"
                )
                if wlogger:
                    wlogger.log_metrics(
                        {
                            "train/total_loss": avg["total"],
                            "train/recon_loss": avg["recon"],
                            "train/kld_loss": avg["kld"],
                            "train/perceptual_loss": avg["perceptual"],
                            "train/gen_loss": avg["gen"],
                            "train/disc_loss": avg["disc"],
                            "train/lr": lr,
                            "train/epoch": epoch + 1,
                            "train/global_step": global_step,
                            "train/samples_per_sec": samples_per_sec,
                        }
                    )

            # save reconstruction samples
            if (
                global_step % args.save_samples_every == 0
                and is_main_process()
                and vis_images is not None
            ):
                _model = model.module if hasattr(model, "module") else model
                sample_path = os.path.join(
                    args.results_dir, "samples", f"recon_step{global_step:07d}.png"
                )
                save_reconstructions(_model, vis_images, sample_path)

        # end of epoch
        if is_main_process():
            avg = {k: v / max(epoch_samples, 1) for k, v in epoch_losses.items()}
            elapsed = time.time() - t0
            print(
                f"--- epoch {epoch+1} done in {elapsed:.0f}s | avg total={avg['total']:.4f} ---"
            )

            # save checkpoint
            _model = model.module if hasattr(model, "module") else model
            ckpt = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "model": _model.state_dict(),
                "optimizer_vae": optimizer_vae.state_dict(),
                "optimizer_disc": optimizer_disc.state_dict(),
                "scaler": scaler.state_dict(),
                "args": vars(args),
            }
            ckpt_path = os.path.join(
                args.results_dir, "checkpoints", f"epoch{epoch+1:04d}.pt"
            )
            save_checkpoint(ckpt, ckpt_path)
            # also save latest
            save_checkpoint(
                ckpt, os.path.join(args.results_dir, "checkpoints", "latest.pt")
            )
            print(f"saved checkpoint: {ckpt_path}")

        # generate fid samples at end of training or every N epochs
        if (epoch + 1) % args.fid_every == 0 or (epoch + 1) == args.epochs:
            if is_main_process():
                _model = model.module if hasattr(model, "module") else model
                npz_path = os.path.join(
                    args.results_dir, "samples", f"gen_epoch{epoch+1:04d}.npz"
                )
                n = save_samples_npz(
                    _model,
                    args.fid_num_samples,
                    args.latent_dim,
                    args.distribution,
                    device,
                    npz_path,
                )
                print(f"saved {n} generated samples to {npz_path}")

    cleanup_distributed()
    if is_main_process():
        print("training complete.")


def main():
    parser = argparse.ArgumentParser(
        description="imagenet 256 training for clifford vae"
    )

    # data
    parser.add_argument(
        "--data-path", type=str, required=True, help="path to imagenet tar or folder"
    )
    parser.add_argument("--results-dir", type=str, default="results/imagenet")
    parser.add_argument("--image-size", type=int, default=256)

    # model
    parser.add_argument("--latent-dim", type=int, default=16)
    parser.add_argument(
        "--patch-size", type=int, default=0, help="0 = use default for image size"
    )
    parser.add_argument(
        "--distribution",
        type=str,
        default="clifford",
        choices=["clifford", "powerspherical", "gaussian"],
    )

    # training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--global-batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=5000)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument(
        "--use-amp", action="store_true", default=True, help="bf16 mixed precision"
    )
    parser.add_argument("--no-amp", dest="use_amp", action="store_false")
    parser.add_argument("--num-workers", type=int, default=8)

    # loss weights
    parser.add_argument("--kl-weight", type=float, default=0.004)
    parser.add_argument("--perceptual-weight", type=float, default=1.0)
    parser.add_argument("--reconstruction-weight", type=float, default=1.0)
    parser.add_argument("--adversarial-weight", type=float, default=0.1)
    parser.add_argument(
        "--disc-start-step",
        type=int,
        default=50000,
        help="start discriminator training after N steps",
    )

    # logging
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--save-samples-every", type=int, default=2000)
    parser.add_argument(
        "--fid-every", type=int, default=10, help="generate fid samples every N epochs"
    )
    parser.add_argument("--fid-num-samples", type=int, default=50000)
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--wandb-project", type=str, default="clifford-vae-imagenet")

    # checkpointing
    parser.add_argument(
        "--resume", type=str, default=None, help="path to checkpoint to resume from"
    )

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()

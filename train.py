# -*- coding: utf-8 -*-

"""Training script for RTM-UIE."""

# -*- coding: utf-8 -*-

import os
import glob
import json
import random
import argparse
from dataclasses import dataclass, asdict
from typing import List, Tuple

import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import make_grid, save_image
from tqdm import tqdm

from retinex.decomp.retinex_decomposer import RetinexDecomposer

EXTS = [".png", ".jpg", ".jpeg", ".bmp", ".PNG", ".JPG", ".JPEG", ".BMP"]

def list_images(root: str) -> List[str]:
    paths = []
    for e in EXTS:
        paths += glob.glob(os.path.join(root, f"*{e}"))
    return sorted(paths)

def stem(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]

def find_gt(gt_root: str, s: str) -> str:
    for e in EXTS:
        p = os.path.join(gt_root, s + e)
        if os.path.exists(p):
            return p
    return ""

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def make_pairs(input_dir: str, gt_dir: str, tag: str) -> List[Tuple[str, str, str]]:
    pairs = []
    in_paths = list_images(input_dir)
    if len(in_paths) == 0:
        raise RuntimeError(f"No images found in {input_dir}")

    missing = 0
    for ip in in_paths:
        s = stem(ip)
        gp = find_gt(gt_dir, s)
        if not gp:
            print(f"[WARN] GT missing for {tag}: {ip}")
            missing += 1
            continue
        pairs.append((ip, gp, tag))

    if len(pairs) == 0:
        raise RuntimeError(f"No valid pairs found for {tag}: {input_dir} / {gt_dir}")

    print(f"[{tag}] valid pairs: {len(pairs)}, missing: {missing}")
    return pairs

def split_pairs(pairs: List[Tuple[str, str, str]], train_ratio: float, seed: int):
    idxs = list(range(len(pairs)))
    rng = random.Random(seed)
    rng.shuffle(idxs)
    split = int(len(idxs) * train_ratio)
    train_pairs = [pairs[i] for i in idxs[:split]]
    val_pairs = [pairs[i] for i in idxs[split:]]
    return train_pairs, val_pairs

def pad_to_multiple(x: torch.Tensor, m: int = 16):
    _, _, h, w = x.shape
    pad_h = (m - h % m) % m
    pad_w = (m - w % m) % m
    x_pad = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    return x_pad, pad_h, pad_w

def unpad(x: torch.Tensor, pad_h: int, pad_w: int):
    if pad_h > 0:
        x = x[:, :, :-pad_h, :]
    if pad_w > 0:
        x = x[:, :, :, :-pad_w]
    return x

def rgb_to_luma(x: torch.Tensor) -> torch.Tensor:
    return 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]

def normalize_per_sample(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    b = x.size(0)
    flat = x.view(b, -1)
    x_min = flat.min(dim=1)[0].view(b, 1, 1, 1)
    x_max = flat.max(dim=1)[0].view(b, 1, 1, 1)
    return (x - x_min) / (x_max - x_min + eps)

def tensor_to_np01_rgb(x: torch.Tensor) -> np.ndarray:
    x = x.detach().cpu().clamp(0.0, 1.0)[0]
    return x.permute(1, 2, 0).numpy().astype(np.float32)

def psnr_torch(x: torch.Tensor, y: torch.Tensor) -> float:
    mse = torch.mean((x - y) ** 2)
    if mse.item() <= 1e-12:
        return 100.0
    return float((10.0 * torch.log10(1.0 / mse)).item())

def ssim_torch(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    mu_x = F.avg_pool2d(x, 11, 1, 0)
    mu_y = F.avg_pool2d(y, 11, 1, 0)
    sigma_x = F.avg_pool2d(x * x, 11, 1, 0) - mu_x ** 2
    sigma_y = F.avg_pool2d(y * y, 11, 1, 0) - mu_y ** 2
    sigma_xy = F.avg_pool2d(x * y, 11, 1, 0) - mu_x * mu_y
    num = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    den = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    return (num / den).mean()

def gradient_loss(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    pred_dx = torch.abs(pred[:, :, :, 1:] - pred[:, :, :, :-1])
    pred_dy = torch.abs(pred[:, :, 1:, :] - pred[:, :, :-1, :])
    gt_dx = torch.abs(gt[:, :, :, 1:] - gt[:, :, :, :-1])
    gt_dy = torch.abs(gt[:, :, 1:, :] - gt[:, :, :-1, :])
    return torch.mean(torch.abs(pred_dx - gt_dx)) + torch.mean(torch.abs(pred_dy - gt_dy))

def tv_loss(x: torch.Tensor) -> torch.Tensor:
    dx = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).mean()
    dy = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).mean()
    return dx + dy

class MixedPairedPatchDataset(Dataset):
    def __init__(self, pairs: List[Tuple[str, str, str]], patch: int = 256):
        self.pairs = pairs
        self.patch = patch
        self.t = transforms.ToTensor()

    def __len__(self):
        return len(self.pairs)

    def _resize_min_side(self, im: Image.Image) -> Image.Image:
        w, h = im.size
        p = self.patch
        if min(w, h) >= p:
            return im
        if w < h:
            new_w = p
            new_h = int(h * (p / w))
        else:
            new_h = p
            new_w = int(w * (p / h))
        return im.resize((new_w, new_h), Image.BICUBIC)

    def __getitem__(self, idx):
        ip, gp, tag = self.pairs[idx]
        i_img = Image.open(ip).convert("RGB")
        gt_img = Image.open(gp).convert("RGB")
        i_img = self._resize_min_side(i_img)
        gt_img = self._resize_min_side(gt_img)
        if i_img.size != gt_img.size:
            gt_img = gt_img.resize(i_img.size, Image.BICUBIC)
        w, h = i_img.size
        p = self.patch
        x0 = random.randint(0, w - p)
        y0 = random.randint(0, h - p)
        i_img = i_img.crop((x0, y0, x0 + p, y0 + p))
        gt_img = gt_img.crop((x0, y0, x0 + p, y0 + p))
        if random.random() < 0.5:
            i_img = i_img.transpose(Image.FLIP_LEFT_RIGHT)
            gt_img = gt_img.transpose(Image.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            i_img = i_img.transpose(Image.FLIP_TOP_BOTTOM)
            gt_img = gt_img.transpose(Image.FLIP_TOP_BOTTOM)
        if random.random() < 0.5:
            k = random.choice([1, 2, 3])
            i_img = i_img.rotate(90 * k, expand=True)
            gt_img = gt_img.rotate(90 * k, expand=True)
        return self.t(i_img), self.t(gt_img), tag

class FrozenRetinexFrontend(nn.Module):
    def __init__(self, ckpt_path: str, device: torch.device):
        super().__init__()
        self.decomposer = RetinexDecomposer(base=32, use_refine=False).to(device)
        state = torch.load(ckpt_path, map_location=device)
        if isinstance(state, dict) and "model" in state:
            state = state["model"]
        self.decomposer.load_state_dict(state)
        self.decomposer.eval()
        for p in self.decomposer.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor):
        l, r = self.decomposer(x)
        l = torch.clamp(l, 0.0, 1.0)
        r = torch.clamp(r, 0.0, 1.0)
        if l.size(1) == 1:
            l = l.repeat(1, 3, 1, 1)
        if r.size(1) == 1:
            r = r.repeat(1, 3, 1, 1)
        i0 = torch.clamp(l * r, 0.0, 1.0)
        return l, r, i0

def local_contrast_map(i_img: torch.Tensor) -> torch.Tensor:
    y = rgb_to_luma(i_img)
    dx = torch.abs(y[:, :, :, 1:] - y[:, :, :, :-1])
    dy = torch.abs(y[:, :, 1:, :] - y[:, :, :-1, :])
    dx = F.pad(dx, (0, 1, 0, 0), mode="replicate")
    dy = F.pad(dy, (0, 0, 0, 1), mode="replicate")
    g = dx + dy
    g = F.avg_pool2d(g, kernel_size=7, stride=1, padding=3)
    return normalize_per_sample(g)

def saturation_map(i_img: torch.Tensor) -> torch.Tensor:
    cmax = i_img.max(dim=1, keepdim=True)[0]
    cmin = i_img.min(dim=1, keepdim=True)[0]
    sat = (cmax - cmin) / (cmax + 1e-6)
    sat = F.avg_pool2d(sat, kernel_size=7, stride=1, padding=3)
    return normalize_per_sample(sat)

def build_t_prior_v2(i_img: torch.Tensor, t_min: float = 0.05, alpha: float = 0.6) -> torch.Tensor:
    c_map = local_contrast_map(i_img)
    s_map = saturation_map(i_img)
    t_prior = alpha * c_map + (1.0 - alpha) * s_map
    t_prior = F.avg_pool2d(t_prior, kernel_size=5, stride=1, padding=2)
    return torch.clamp(t_prior, t_min, 1.0)

@torch.no_grad()
def estimate_global_A(x: torch.Tensor, top_ratio: float = 0.001) -> torch.Tensor:
    b, c, h, w = x.shape
    luma = rgb_to_luma(x).reshape(b, -1)
    x_flat = x.reshape(b, c, -1)
    n = luma.size(1)
    k = max(1, int(n * top_ratio))
    a_list = []
    for i in range(b):
        _, idx = torch.topk(luma[i], k=k, largest=True, sorted=False)
        vals = x_flat[i, :, idx]
        a = vals.mean(dim=1, keepdim=True).unsqueeze(-1)
        a_list.append(a)
    return torch.stack(a_list, dim=0)

def build_B_from_tA(t: torch.Tensor, A: torch.Tensor) -> torch.Tensor:
    return A * (1.0 - t)

class TransmissionNet(nn.Module):
    def __init__(self, in_ch: int = 6, base: int = 16, t_min: float = 0.05):
        super().__init__()
        self.t_min = t_min
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, base, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, base, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, 1, 1, 1, 0),
        )

    def forward(self, x: torch.Tensor):
        raw_t = self.net(x)
        t = self.t_min + (1.0 - self.t_min) * torch.sigmoid(raw_t)
        t = F.avg_pool2d(t, kernel_size=5, stride=1, padding=2)
        return t

class ConvAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.act(self.conv(x))

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1 = ConvAct(ch, ch)
        self.c2 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.act = nn.LeakyReLU(0.1, inplace=True)

    def forward(self, x):
        return self.act(x + self.c2(self.c1(x)))

class StrongRefineNetT(nn.Module):
    def __init__(self, in_ch=9, base=48, residual_base=0.10, residual_gamma=0.45):
        super().__init__()
        self.residual_base = residual_base
        self.residual_gamma = residual_gamma

        self.e1 = nn.Sequential(ConvAct(in_ch, base), ResBlock(base), ResBlock(base))
        self.d1 = nn.Conv2d(base, base * 2, 4, 2, 1)
        self.e2 = nn.Sequential(ConvAct(base * 2, base * 2), ResBlock(base * 2), ResBlock(base * 2))
        self.d2 = nn.Conv2d(base * 2, base * 4, 4, 2, 1)
        self.e3 = nn.Sequential(ConvAct(base * 4, base * 4), ResBlock(base * 4), ResBlock(base * 4))
        self.d3 = nn.Conv2d(base * 4, base * 4, 4, 2, 1)
        self.mid = nn.Sequential(ConvAct(base * 4, base * 4), ResBlock(base * 4), ResBlock(base * 4), ResBlock(base * 4))

        self.u3 = nn.Conv2d(base * 8, base * 4, 3, 1, 1)
        self.dec3 = nn.Sequential(ConvAct(base * 4, base * 4), ResBlock(base * 4))
        self.u2 = nn.Conv2d(base * 6, base * 2, 3, 1, 1)
        self.dec2 = nn.Sequential(ConvAct(base * 2, base * 2), ResBlock(base * 2))
        self.u1 = nn.Conv2d(base * 3, base, 3, 1, 1)
        self.dec1 = nn.Sequential(ConvAct(base, base), ResBlock(base), ResBlock(base))

        self.out_delta = nn.Conv2d(base, 3, 1)

    def forward(self, x: torch.Tensor, base_img: torch.Tensor, t_map: torch.Tensor):
        e1 = self.e1(x)
        e2 = self.e2(self.d1(e1))
        e3 = self.e3(self.d2(e2))
        m = self.mid(self.d3(e3))

        m = F.interpolate(m, size=e3.shape[-2:], mode="bilinear", align_corners=False)
        y = self.dec3(self.u3(torch.cat([m, e3], dim=1)))
        y = F.interpolate(y, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        y = self.dec2(self.u2(torch.cat([y, e2], dim=1)))
        y = F.interpolate(y, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        y = self.dec1(self.u1(torch.cat([y, e1], dim=1)))

        delta = torch.tanh(self.out_delta(y))
        gate = self.residual_base + self.residual_gamma * (1.0 - t_map)
        gate = torch.clamp(gate, 0.0, 1.0)
        pred = torch.clamp(base_img + gate.repeat(1, 3, 1, 1) * delta, 0.0, 1.0)
        return pred, delta, gate

@dataclass
class TrainCfg:
    uieb_input_dir: str = "data/Train_UIEB/input"
    uieb_gt_dir: str = "data/Train_UIEB/GT"
    lsui_input_dir: str = "data/Train_LSUI/input"
    lsui_gt_dir: str = "data/Train_LSUI/GT"
    out_dir: str = "runs/rtm_uie_strong_fair_e40"
    retinex_ckpt: str = "pretrained/causal_retinex_best_epoch049_loss0.0065.pth"
    init_tnet_ckpt: str = ""
    init_ckpt: str = ""

    seed: int = 123
    train_ratio: float = 0.90
    patch: int = 256
    batch: int = 6
    lr: float = 1e-4
    tnet_lr: float = 2e-5
    epochs: int = 40
    num_workers: int = 4
    amp: bool = True

    t_base: int = 16
    t_min: float = 0.05
    prior_alpha: float = 0.6
    strong_base: int = 48
    residual_base: float = 0.10
    residual_gamma: float = 0.45

    repeat_uieb_train: int = 4
    repeat_lsui_train: int = 1

    lam_l1: float = 1.0
    lam_mse: float = 0.25
    lam_ssim: float = 1.0
    lam_grad: float = 0.05
    lam_t_tv: float = 0.01
    lam_t_prior: float = 0.05

    score_uieb_weight: float = 0.50
    score_lsui_weight: float = 0.50
    score_ssim_weight: float = 100.0

    val_every: int = 2
    vis_every: int = 300
    freeze_tnet: bool = True
    ablation: str = "full"

def load_tnet_from_ckpt(path: str, tnet: TransmissionNet, device: torch.device):
    if not path:
        print("[Init] No init_tnet_ckpt. tnet starts from scratch.")
        return
    if not os.path.exists(path):
        raise RuntimeError(f"init_tnet_ckpt not found: {path}")
    state = torch.load(path, map_location=device)
    if "tnet" not in state:
        raise RuntimeError(f"init_tnet_ckpt must contain key 'tnet'. Got keys={list(state.keys())}")
    tnet.load_state_dict(state["tnet"])
    print("[Init] loaded tnet from:", path)

def get_ablation_name(cfg_like):
    if cfg_like is None:
        return "full"
    return getattr(cfg_like, "ablation", "full")

def _apply_ablation_after_t(i_img, i0, t_map, b_map, ablation):
    if ablation == "no_B":
        b_map = torch.zeros_like(i_img)

    t_for_gate = t_map
    if ablation == "no_modulation":
        t_for_gate = torch.ones_like(t_map)

    return i0, t_map, b_map, t_for_gate

@torch.no_grad()
def forward_model(retinex_front, tnet, refine_net, i_img, cfg_like=None):
    ablation = get_ablation_name(cfg_like)

    _, _, i0 = retinex_front(i_img)

    if ablation == "no_retinex":
        i0 = i_img

    t_map = tnet(torch.cat([i_img, i0], dim=1))
    a_map = estimate_global_A(i_img)
    b_map = build_B_from_tA(t_map, a_map)

    i0, t_map, b_map, t_for_gate = _apply_ablation_after_t(
        i_img, i0, t_map, b_map, ablation
    )

    x = torch.cat([i_img, i0, b_map], dim=1)
    if x.size(1) != 9:
        raise RuntimeError(f"Refinement input must be 9 channels, got {x.size(1)}.")

    pred, delta, gate = refine_net(x, base_img=i0, t_map=t_for_gate)
    return pred, i0, t_map, b_map, gate

def forward_ablation_train(retinex_front, tnet, refine_net, i_img, cfg_like):
    ablation = get_ablation_name(cfg_like)

    with torch.no_grad():
        _, _, i0 = retinex_front(i_img)

    if ablation == "no_retinex":
        i0 = i_img

    t_map = tnet(torch.cat([i_img, i0], dim=1))
    a_map = estimate_global_A(i_img)
    b_map = build_B_from_tA(t_map, a_map)

    i0, t_map, b_map, t_for_gate = _apply_ablation_after_t(
        i_img, i0, t_map, b_map, ablation
    )

    x = torch.cat([i_img, i0, b_map], dim=1)
    if x.size(1) != 9:
        raise RuntimeError(f"Refinement input must be 9 channels, got {x.size(1)}.")

    pred, delta, gate = refine_net(x, base_img=i0, t_map=t_for_gate)
    return pred, i0, t_map, b_map, gate

@torch.no_grad()
def val_full(retinex_front, tnet, refine_net, val_pairs, device, cfg_like=None):
    retinex_front.eval()
    tnet.eval()
    refine_net.eval()
    tfm = transforms.ToTensor()
    psnrs, ssims = [], []
    for ip, gp, _tag in tqdm(val_pairs, desc="val", ncols=120, leave=False):
        i_img = tfm(Image.open(ip).convert("RGB")).unsqueeze(0).to(device)
        gt_img = tfm(Image.open(gp).convert("RGB")).unsqueeze(0).to(device)
        if i_img.shape[-2:] != gt_img.shape[-2:]:
            gt_img = F.interpolate(gt_img, size=i_img.shape[-2:], mode="bilinear", align_corners=False)
        i_pad, ph, pw = pad_to_multiple(i_img, 16)
        pred_pad, _, _, _, _ = forward_model(retinex_front, tnet, refine_net, i_pad, cfg_like)
        pred = unpad(pred_pad, ph, pw).clamp(0.0, 1.0)
        psnrs.append(psnr_torch(pred, gt_img))
        ssims.append(float(ssim_torch(pred, gt_img).item()))
    refine_net.train()
    if len(psnrs) == 0:
        return 0.0, 0.0
    return sum(psnrs) / len(psnrs), sum(ssims) / len(ssims)

def calc_score(psnr: float, ssim: float, weight: float = 100.0) -> float:
    return psnr + weight * ssim

def save_checkpoint(path, tnet, refine_net, epoch, metrics, cfg, tag):
    torch.save(
        {
            "tnet": tnet.state_dict(),
            "refine_net": refine_net.state_dict(),
            "epoch": epoch,
            "metrics": metrics,
            "cfg": asdict(cfg),
            "tag": tag,
            "model_type": "StrongRefineNetT_9ch",
        },
        path,
    )

def train_main(args):
    cfg = TrainCfg()
    cfg.out_dir = args.out_dir or cfg.out_dir
    cfg.retinex_ckpt = args.retinex_ckpt or cfg.retinex_ckpt
    cfg.init_tnet_ckpt = args.init_tnet_ckpt or cfg.init_tnet_ckpt
    cfg.init_ckpt = args.init_ckpt or cfg.init_ckpt
    cfg.epochs = args.epochs if args.epochs is not None else cfg.epochs
    cfg.batch = args.batch if args.batch is not None else cfg.batch
    cfg.lr = args.lr if args.lr is not None else cfg.lr
    cfg.tnet_lr = args.tnet_lr if args.tnet_lr is not None else cfg.tnet_lr
    cfg.val_every = args.val_every if args.val_every is not None else cfg.val_every
    cfg.num_workers = args.num_workers if args.num_workers is not None else cfg.num_workers
    cfg.repeat_uieb_train = args.repeat_uieb_train if args.repeat_uieb_train is not None else cfg.repeat_uieb_train
    cfg.repeat_lsui_train = args.repeat_lsui_train if args.repeat_lsui_train is not None else cfg.repeat_lsui_train
    cfg.lam_l1 = args.lam_l1 if args.lam_l1 is not None else cfg.lam_l1
    cfg.lam_mse = args.lam_mse if args.lam_mse is not None else cfg.lam_mse
    cfg.lam_ssim = args.lam_ssim if args.lam_ssim is not None else cfg.lam_ssim
    cfg.lam_grad = args.lam_grad if args.lam_grad is not None else cfg.lam_grad
    cfg.lam_t_tv = args.lam_t_tv if args.lam_t_tv is not None else cfg.lam_t_tv
    cfg.lam_t_prior = args.lam_t_prior if args.lam_t_prior is not None else cfg.lam_t_prior
    cfg.freeze_tnet = args.freeze_tnet
    cfg.ablation = args.ablation
    cfg.amp = not args.no_amp

    ensure_dir(cfg.out_dir)
    ensure_dir(os.path.join(cfg.out_dir, "vis"))
    with open(os.path.join(cfg.out_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump(asdict(cfg), f, indent=2, ensure_ascii=False)

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.cuda.manual_seed_all(cfg.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("device:", device)
    print("out_dir:", cfg.out_dir)
    print("retinex_ckpt:", cfg.retinex_ckpt)
    print("init_tnet_ckpt:", cfg.init_tnet_ckpt if cfg.init_tnet_ckpt else "<none>")
    print("init_ckpt:", cfg.init_ckpt if cfg.init_ckpt else "<none>")
    print("epochs:", cfg.epochs, "batch:", cfg.batch, "lr:", cfg.lr, "tnet_lr:", cfg.tnet_lr)
    print("repeat_uieb_train:", cfg.repeat_uieb_train, "repeat_lsui_train:", cfg.repeat_lsui_train)
    print("freeze_tnet:", cfg.freeze_tnet)
    print("ablation:", cfg.ablation)
    print("loss weights:",
          "l1=", cfg.lam_l1,
          "mse=", cfg.lam_mse,
          "ssim=", cfg.lam_ssim,
          "grad=", cfg.lam_grad,
          "tTV=", cfg.lam_t_tv,
          "tP=", cfg.lam_t_prior)

    uieb_pairs = make_pairs(cfg.uieb_input_dir, cfg.uieb_gt_dir, "Train_UIEB")
    lsui_pairs = make_pairs(cfg.lsui_input_dir, cfg.lsui_gt_dir, "Train_LSUI")
    uieb_train, uieb_val = split_pairs(uieb_pairs, cfg.train_ratio, cfg.seed + 1)
    lsui_train, lsui_val = split_pairs(lsui_pairs, cfg.train_ratio, cfg.seed + 2)
    print(f"[Train_UIEB] train={len(uieb_train)}, val={len(uieb_val)}")
    print(f"[Train_LSUI] train={len(lsui_train)}, val={len(lsui_val)}")

    mixed_train_pairs = uieb_train * cfg.repeat_uieb_train + lsui_train * cfg.repeat_lsui_train
    random.Random(cfg.seed + 3).shuffle(mixed_train_pairs)
    print("[Mixed] train pairs after repeat:", len(mixed_train_pairs))

    train_ds = MixedPairedPatchDataset(mixed_train_pairs, patch=cfg.patch)
    train_dl = DataLoader(train_ds, batch_size=cfg.batch, shuffle=True, num_workers=cfg.num_workers, pin_memory=True, drop_last=True)

    retinex_front = FrozenRetinexFrontend(cfg.retinex_ckpt, device)
    print("[Retinex] loaded and frozen:", cfg.retinex_ckpt)

    tnet = TransmissionNet(in_ch=6, base=cfg.t_base, t_min=cfg.t_min).to(device)
    load_tnet_from_ckpt(cfg.init_tnet_ckpt, tnet, device)

    refine_net = StrongRefineNetT(in_ch=9, base=cfg.strong_base, residual_base=cfg.residual_base, residual_gamma=cfg.residual_gamma).to(device)

    if cfg.init_ckpt:
        if not os.path.exists(cfg.init_ckpt):
            raise RuntimeError(f"init_ckpt not found: {cfg.init_ckpt}")
        full_state = torch.load(cfg.init_ckpt, map_location=device)
        if "tnet" not in full_state or "refine_net" not in full_state:
            raise RuntimeError(f"init_ckpt must contain 'tnet' and 'refine_net'. Got keys={list(full_state.keys())}")
        tnet.load_state_dict(full_state["tnet"])
        refine_net.load_state_dict(full_state["refine_net"])
        print("[Init] loaded full model from:", cfg.init_ckpt)

    if cfg.freeze_tnet:
        for p in tnet.parameters():
            p.requires_grad = False

    param_groups = [{"params": refine_net.parameters(), "lr": cfg.lr}]
    if not cfg.freeze_tnet:
        param_groups.append({"params": tnet.parameters(), "lr": cfg.tnet_lr})
    opt = torch.optim.Adam(param_groups)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp)

    init_uieb_psnr, init_uieb_ssim = val_full(retinex_front, tnet, refine_net, uieb_val, device, cfg)
    init_lsui_psnr, init_lsui_ssim = val_full(retinex_front, tnet, refine_net, lsui_val, device, cfg)
    init_uieb_score = calc_score(init_uieb_psnr, init_uieb_ssim, cfg.score_ssim_weight)
    init_lsui_score = calc_score(init_lsui_psnr, init_lsui_ssim, cfg.score_ssim_weight)
    init_mix = cfg.score_uieb_weight * init_uieb_score + cfg.score_lsui_weight * init_lsui_score
    print("\n[INIT_VAL]")
    print(f"Train_UIEB-val: PSNR={init_uieb_psnr:.2f}, SSIM={init_uieb_ssim:.4f}, SCORE={init_uieb_score:.4f}")
    print(f"Train_LSUI-val: PSNR={init_lsui_psnr:.2f}, SSIM={init_lsui_ssim:.4f}, SCORE={init_lsui_score:.4f}")
    print(f"MIX_SCORE={init_mix:.4f}\n")

    best_mix = init_mix
    best_uieb_psnr = init_uieb_psnr
    best_lsui_psnr = init_lsui_psnr
    best_uieb_score = init_uieb_score
    best_lsui_score = init_lsui_score

    init_metrics = {
        "uieb_val_psnr": init_uieb_psnr,
        "uieb_val_ssim": init_uieb_ssim,
        "uieb_val_score": init_uieb_score,
        "lsui_val_psnr": init_lsui_psnr,
        "lsui_val_ssim": init_lsui_ssim,
        "lsui_val_score": init_lsui_score,
        "mix_score": init_mix,
    }
    save_checkpoint(os.path.join(cfg.out_dir, "init.pth"), tnet, refine_net, -1, init_metrics, cfg, "init")
    save_checkpoint(os.path.join(cfg.out_dir, "best_mix_score.pth"), tnet, refine_net, -1, init_metrics, cfg, "best_mix_init")
    save_checkpoint(os.path.join(cfg.out_dir, "best_uieb_psnr.pth"), tnet, refine_net, -1, init_metrics, cfg, "best_uieb_psnr_init")
    save_checkpoint(os.path.join(cfg.out_dir, "best_lsui_psnr.pth"), tnet, refine_net, -1, init_metrics, cfg, "best_lsui_psnr_init")
    save_checkpoint(os.path.join(cfg.out_dir, "best.pth"), tnet, refine_net, -1, init_metrics, cfg, "best_alias_init")

    log_path = os.path.join(cfg.out_dir, "train_log.csv")
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("epoch,uieb_val_psnr,uieb_val_ssim,uieb_val_score,lsui_val_psnr,lsui_val_ssim,lsui_val_score,mix_score,best_type\n")
        f.write(f"-1,{init_uieb_psnr:.6f},{init_uieb_ssim:.6f},{init_uieb_score:.6f},{init_lsui_psnr:.6f},{init_lsui_ssim:.6f},{init_lsui_score:.6f},{init_mix:.6f},init\n")

    global_step = 0
    for ep in range(cfg.epochs):
        refine_net.train()
        if cfg.freeze_tnet:
            tnet.eval()
        else:
            tnet.train()
        pbar = tqdm(train_dl, ncols=180)
        for i_img, gt_img, tag in pbar:
            i_img = i_img.to(device, non_blocking=True)
            gt_img = gt_img.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=cfg.amp):
                t_prior = build_t_prior_v2(i_img, t_min=cfg.t_min, alpha=cfg.prior_alpha)
                pred, i0, t_map, b_map, gate = forward_ablation_train(
                    retinex_front, tnet, refine_net, i_img, cfg
                )

                l1 = torch.mean(torch.abs(pred - gt_img))
                mse = torch.mean((pred - gt_img) ** 2)
                ssim = ssim_torch(pred, gt_img)
                g = gradient_loss(pred, gt_img)
                t_tv = tv_loss(t_map)
                t_prior_l1 = torch.mean(torch.abs(t_map - t_prior))
                loss = cfg.lam_l1 * l1 + cfg.lam_mse * mse + cfg.lam_ssim * (1.0 - ssim) + cfg.lam_grad * g + cfg.lam_t_tv * t_tv + cfg.lam_t_prior * t_prior_l1

            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            global_step += 1
            pbar.set_description(f"ep{ep:03d} loss={loss.item():.4f} l1={l1.item():.4f} mse={mse.item():.5f} ssim={ssim.item():.4f} g={g.item():.4f} tTV={t_tv.item():.4f} tP={t_prior_l1.item():.4f}")

            if global_step % cfg.vis_every == 0:
                tp_vis = t_prior[:1].detach().cpu().repeat(1, 3, 1, 1)
                t_vis = t_map[:1].detach().cpu().repeat(1, 3, 1, 1)
                b_vis = b_map[:1].detach().cpu()
                gate_vis = gate[:1].detach().cpu().repeat(1, 3, 1, 1)
                panel = torch.cat([i_img[:1].detach().cpu(), i0[:1].detach().cpu(), tp_vis, t_vis, b_vis, gate_vis, pred[:1].detach().cpu(), gt_img[:1].detach().cpu()], dim=0)
                grid = make_grid(panel, nrow=8, padding=2)
                save_image(grid, os.path.join(cfg.out_dir, "vis", f"step{global_step:06d}.png"))

        do_val = (ep == 0) or ((ep + 1) % cfg.val_every == 0) or (ep == cfg.epochs - 1)
        if not do_val:
            continue
        uieb_psnr, uieb_ssim = val_full(retinex_front, tnet, refine_net, uieb_val, device, cfg)
        lsui_psnr, lsui_ssim = val_full(retinex_front, tnet, refine_net, lsui_val, device, cfg)
        uieb_score = calc_score(uieb_psnr, uieb_ssim, cfg.score_ssim_weight)
        lsui_score = calc_score(lsui_psnr, lsui_ssim, cfg.score_ssim_weight)
        mix_score = cfg.score_uieb_weight * uieb_score + cfg.score_lsui_weight * lsui_score
        print("\n[VAL]")
        print(f"ep{ep:03d} Train_UIEB-val: PSNR={uieb_psnr:.2f}, SSIM={uieb_ssim:.4f}, SCORE={uieb_score:.4f}")
        print(f"ep{ep:03d} Train_LSUI-val: PSNR={lsui_psnr:.2f}, SSIM={lsui_ssim:.4f}, SCORE={lsui_score:.4f}")
        print(f"ep{ep:03d} MIX_SCORE={mix_score:.4f}\n")
        metrics = {"uieb_val_psnr": uieb_psnr, "uieb_val_ssim": uieb_ssim, "uieb_val_score": uieb_score, "lsui_val_psnr": lsui_psnr, "lsui_val_ssim": lsui_ssim, "lsui_val_score": lsui_score, "mix_score": mix_score}
        save_checkpoint(os.path.join(cfg.out_dir, "latest.pth"), tnet, refine_net, ep, metrics, cfg, "latest")
        updated = []
        if mix_score > best_mix:
            best_mix = mix_score
            updated.append("best_mix_score")
            save_checkpoint(os.path.join(cfg.out_dir, "best_mix_score.pth"), tnet, refine_net, ep, metrics, cfg, "best_mix_score")
            save_checkpoint(os.path.join(cfg.out_dir, "best.pth"), tnet, refine_net, ep, metrics, cfg, "best_alias")
            print(f"[BEST_MIX_SCORE] updated at ep{ep:03d}: {mix_score:.4f}")
        if uieb_psnr > best_uieb_psnr:
            best_uieb_psnr = uieb_psnr
            updated.append("best_uieb_psnr")
            save_checkpoint(os.path.join(cfg.out_dir, "best_uieb_psnr.pth"), tnet, refine_net, ep, metrics, cfg, "best_uieb_psnr")
            print(f"[BEST_UIEB_PSNR] updated at ep{ep:03d}: PSNR={uieb_psnr:.2f}, SSIM={uieb_ssim:.4f}")
        if lsui_psnr > best_lsui_psnr:
            best_lsui_psnr = lsui_psnr
            updated.append("best_lsui_psnr")
            save_checkpoint(os.path.join(cfg.out_dir, "best_lsui_psnr.pth"), tnet, refine_net, ep, metrics, cfg, "best_lsui_psnr")
            print(f"[BEST_LSUI_PSNR] updated at ep{ep:03d}: PSNR={lsui_psnr:.2f}, SSIM={lsui_ssim:.4f}")
        if uieb_score > best_uieb_score:
            best_uieb_score = uieb_score
            updated.append("best_uieb_score")
        if lsui_score > best_lsui_score:
            best_lsui_score = lsui_score
            updated.append("best_lsui_score")
        if not updated:
            updated = ["none"]
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(f"{ep},{uieb_psnr:.6f},{uieb_ssim:.6f},{uieb_score:.6f},{lsui_psnr:.6f},{lsui_ssim:.6f},{lsui_score:.6f},{mix_score:.6f},{'|'.join(updated)}\n")

    print("Training done.")
    print("Output dir:", cfg.out_dir)
    print("Best mix score ckpt:", os.path.join(cfg.out_dir, "best_mix_score.pth"))
    print("Best UIEB PSNR ckpt:", os.path.join(cfg.out_dir, "best_uieb_psnr.pth"))
    print("Best LSUI PSNR ckpt:", os.path.join(cfg.out_dir, "best_lsui_psnr.pth"))
    print("Best alias ckpt:", os.path.join(cfg.out_dir, "best.pth"))
    print("Train log:", log_path)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ablation",
        type=str,
        default="full",
        choices=["full", "no_retinex", "no_B", "no_modulation"],
        help="leave-one-out ablation setting",
    )

    parser.add_argument("--out_dir", type=str, default=None)
    parser.add_argument("--retinex_ckpt", type=str, default=None)
    parser.add_argument("--init_tnet_ckpt", type=str, default=None)
    parser.add_argument("--init_ckpt", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--tnet_lr", type=float, default=None)
    parser.add_argument("--val_every", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--repeat_uieb_train", type=int, default=None)
    parser.add_argument("--repeat_lsui_train", type=int, default=None)
    parser.add_argument("--lam_l1", type=float, default=None)
    parser.add_argument("--lam_mse", type=float, default=None)
    parser.add_argument("--lam_ssim", type=float, default=None)
    parser.add_argument("--lam_grad", type=float, default=None)
    parser.add_argument("--lam_t_tv", type=float, default=None)
    parser.add_argument("--lam_t_prior", type=float, default=None)
    parser.add_argument("--residual_base", type=float, default=None)
    parser.add_argument("--residual_gamma", type=float, default=None)
    parser.add_argument("--freeze_tnet", action="store_true")
    parser.add_argument("--no_amp", action="store_true")

    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train_main(args)

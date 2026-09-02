# -*- coding: utf-8 -*-

"""Inference script for RTM-UIE."""

import argparse
import json
import os

from PIL import Image
import torch
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

from train import (
    list_images,
    stem,
    ensure_dir,
    pad_to_multiple,
    unpad,
    FrozenRetinexFrontend,
    TransmissionNet,
    StrongRefineNetT,
    estimate_global_A,
    build_B_from_tA,
)

@torch.no_grad()
def enhance_image(image_path, retinex_front, tnet, refine_net, device):
    transform = transforms.ToTensor()
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)

    image_pad, pad_h, pad_w = pad_to_multiple(image, 16)

    _, _, retinex_prior = retinex_front(image_pad)
    transmission = tnet(torch.cat([image_pad, retinex_prior], dim=1))
    atmospheric_light = estimate_global_A(image_pad)
    scattering_cue = build_B_from_tA(transmission, atmospheric_light)

    refine_input = torch.cat([image_pad, retinex_prior, scattering_cue], dim=1)
    if refine_input.size(1) != 9:
        raise RuntimeError(
            f"Refinement input must be 9 channels, got {refine_input.size(1)}."
        )

    pred_pad, _, _ = refine_net(
        refine_input,
        base_img=retinex_prior,
        t_map=transmission,
    )

    pred = unpad(pred_pad, pad_h, pad_w).clamp(0.0, 1.0)
    return pred

def load_models(args, device):
    state = torch.load(args.ckpt, map_location=device)
    cfg = state.get("cfg", {})

    retinex_ckpt = args.retinex_ckpt or cfg.get(
        "retinex_ckpt",
        "pretrained/causal_retinex_best_epoch049_loss0.0065.pth",
    )

    t_base = int(cfg.get("t_base", 16))
    t_min = float(cfg.get("t_min", 0.05))
    strong_base = int(cfg.get("strong_base", 48))

    residual_base = (
        float(args.residual_base)
        if args.residual_base is not None
        else float(cfg.get("residual_base", 0.10))
    )
    residual_gamma = (
        float(args.residual_gamma)
        if args.residual_gamma is not None
        else float(cfg.get("residual_gamma", 0.45))
    )

    retinex_front = FrozenRetinexFrontend(retinex_ckpt, device)
    tnet = TransmissionNet(in_ch=6, base=t_base, t_min=t_min).to(device)
    refine_net = StrongRefineNetT(
        in_ch=9,
        base=strong_base,
        residual_base=residual_base,
        residual_gamma=residual_gamma,
    ).to(device)

    if "tnet" not in state or "refine_net" not in state:
        raise RuntimeError("Checkpoint must contain 'tnet' and 'refine_net'.")

    tnet.load_state_dict(state["tnet"])
    refine_net.load_state_dict(state["refine_net"])

    retinex_front.eval()
    tnet.eval()
    refine_net.eval()

    return retinex_front, tnet, refine_net, retinex_ckpt

def parse_args():
    parser = argparse.ArgumentParser(description="RTM-UIE inference")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to trained checkpoint.")
    parser.add_argument("--input_dir", type=str, required=True, help="Directory of input images.")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory for enhanced outputs.")
    parser.add_argument("--retinex_ckpt", type=str, default=None, help="Path to Retinex checkpoint.")
    parser.add_argument("--residual_base", type=float, default=None)
    parser.add_argument("--residual_gamma", type=float, default=None)
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ensure_dir(args.out_dir)
    image_paths = list_images(args.input_dir)
    if len(image_paths) == 0:
        raise RuntimeError(f"No images found in {args.input_dir}")

    retinex_front, tnet, refine_net, retinex_ckpt = load_models(args, device)

    config = {
        "checkpoint": args.ckpt,
        "retinex_checkpoint": retinex_ckpt,
        "input_dir": args.input_dir,
        "out_dir": args.out_dir,
        "num_images": len(image_paths),
    }
    with open(os.path.join(args.out_dir, "inference_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, ensure_ascii=False)

    for image_path in tqdm(image_paths, ncols=100):
        pred = enhance_image(
            image_path=image_path,
            retinex_front=retinex_front,
            tnet=tnet,
            refine_net=refine_net,
            device=device,
        )
        save_image(pred[0], os.path.join(args.out_dir, f"{stem(image_path)}.png"))

    print("Inference finished.")
    print("Input dir :", args.input_dir)
    print("Output dir:", args.out_dir)
    print("Images    :", len(image_paths))

if __name__ == "__main__":
    main()

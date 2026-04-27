#!/usr/bin/env python
"""Check _normal field in all submap ckpts."""
import os, sys, argparse, torch, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.registration_2dgs import resolve_2dgs_normals

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--submaps_dir", required=True)
    args = p.parse_args()

    ckpts = sorted(f for f in os.listdir(args.submaps_dir) if f.endswith(".ckpt"))
    for f in ckpts:
        path = os.path.join(args.submaps_dir, f)
        ckpt = torch.load(path, map_location="cpu")
        gp = ckpt.get("gaussian_params", ckpt)
        xyz = gp.get("_xyz")
        opacity = gp.get("_opacity")
        normal, source = resolve_2dgs_normals(gp)

        n_xyz = xyz.shape[0] if xyz is not None else 0
        n_op = opacity.shape[0] if opacity is not None else 0
        n_normal = normal.shape[0] if normal is not None else 0
        nm_mean = normal.norm(dim=-1).mean().item() if normal is not None else 0
        invalid = (normal.norm(dim=-1) < 1e-6).sum().item() if normal is not None else 0

        print(f"{f}: N_xyz={n_xyz} N_opacity={n_op} N_normal={n_normal} "
              f"source={source} norm_mean={nm_mean:.4f} invalid={invalid}")


if __name__ == "__main__":
    main()

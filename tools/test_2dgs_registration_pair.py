#!/usr/bin/env python
"""Unit test: registration_2dgs on a pair of submap ckpts."""
import os, sys, argparse, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.registration_2dgs import load_submap_from_ckpt, registration_2dgs

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src_ckpt", required=True)
    p.add_argument("--tgt_ckpt", required=True)
    p.add_argument("--method", default="2dgs_hybrid")
    args = p.parse_args()

    src = load_submap_from_ckpt(args.src_ckpt, submap_id=0)
    tgt = load_submap_from_ckpt(args.tgt_ckpt, submap_id=1)
    print(f"Source: N={len(src.xyz)} normal_source={src.normal_source}")
    print(f"Target: N={len(tgt.xyz)} normal_source={tgt.normal_source}")

    r = registration_2dgs(src, tgt, np.eye(4), mode="loop")
    print(f"\nsuccess: {r['successful']}")
    print(f"fitness: {r['fitness']:.4f}")
    print(f"rmse: {r['inlier_rmse']:.4f}")
    print(f"normal_score: {r.get('normal_score',0):.4f}")
    print(f"overlap: {r.get('overlap',0):.4f}")
    print(f"reason: {r.get('reason','')}")
    print(f"T:\n{r['transformation']}")


if __name__ == "__main__":
    main()

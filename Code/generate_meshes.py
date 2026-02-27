#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generate a family of meshes for a list of global mesh sizes h.

This script only generates meshes (Gmsh -> XDMF) and stores them under

  mesh/out_h_{h}/mesh.msh
  mesh/out_h_{h}/mesh_cells.xdmf
  mesh/out_h_{h}/mesh_facets.xdmf

It intentionally does NOT solve the optimal control problem.

Typical usage:
  python Code/generate_meshes.py --h-list 0.06 0.04 0.03 0.02 0.015 0.012 0.01 0.008

Then run the solver/plotter (reusing those meshes):
  python Code/mesh_independence.py --no-mesh-gen --h-list ...
"""

import argparse
import os
import sys
from typing import Iterable


def _h_str(h: float) -> str:
    # Match Python's default float -> str behavior used elsewhere.
    return str(float(h))


def iter_h_list(values: Iterable[float]):
    for h in values:
        yield float(h)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate meshes for a list of global sizes h.")
    ap.add_argument(
        "--h-list",
        type=float,
        nargs="+",
        default=[0.06, 0.04, 0.03, 0.02, 0.015, 0.012, 0.01, 0.008],
        help="List of global mesh size h values",
    )
    ap.add_argument("--cx", type=float, default=0.5, help="Circle center x")
    ap.add_argument("--cy", type=float, default=0.5, help="Circle center y")
    ap.add_argument("--r", type=float, default=0.1, help="Circle radius")
    ap.add_argument("--no-refine-circle", action="store_true", help="Disable circle-near refinement")
    ap.add_argument("--mesh-root", type=str, default="mesh", help="Root directory for mesh outputs")
    ap.add_argument(
        "--plots-dir",
        type=str,
        default=None,
        help=(
            "Optional directory for mesh preview images. "
            "If omitted, a per-mesh mesh.png is written into mesh/mesh_h_{h}/"
        ),
    )
    args = ap.parse_args()

    # Local import so non-mesh tasks don't require gmsh.
    # Support both invocation styles:
    #   python -m Code.generate_meshes
    #   python Code/generate_meshes.py
    try:
        from .gmsh_mesh import build_mesh  # type: ignore
    except Exception:
        # When running as a script, sys.path typically contains Code/, so this works:
        try:
            from gmsh_mesh import build_mesh  # type: ignore
        except Exception:
            # Last resort: add repo root and import as package.
            repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
            if repo_root not in sys.path:
                sys.path.insert(0, repo_root)
            from Code.gmsh_mesh import build_mesh  # type: ignore

    mesh_root = args.mesh_root
    os.makedirs(mesh_root, exist_ok=True)

    for h in iter_h_list(args.h_list):
        h_dir = _h_str(h)
        outdir = os.path.join(mesh_root, f"mesh_h_{h_dir}")
        # Avoid overwriting a single global plots/mesh.png by default.
        plots_dir = args.plots_dir if args.plots_dir is not None else outdir

        print(f"Generating mesh for h={h_dir} -> {outdir}")
        build_mesh(
            cx=args.cx,
            cy=args.cy,
            r=args.r,
            h=h,
            refine_circle=(not args.no_refine_circle),
            outdir=outdir,
            plots_dir=plots_dir,
        )


if __name__ == "__main__":
    main()

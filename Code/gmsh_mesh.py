#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Build a unit-square triangular mesh with an exact circular subdomain using Gmsh,
then convert to XDMF with subdomain (cell) and facet tags via meshio.

Outputs under mesh/out/ by default:
- mesh.msh (Gmsh)
- mesh_cells.xdmf  (triangles + subdomain tags)
- mesh_facets.xdmf (lines + boundary tags)
Additionally saves a mesh PNG under plots/ by default.

Physical IDs (2D):
- 1: domain (full square)
- 2: omega (circular subregion)
Physical IDs (1D facets):
- 11: outer boundary
- 12: circle boundary
"""
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


def build_mesh(cx=0.5, cy=0.5, r=0.1, h=0.03, refine_circle=True, outdir="mesh/out", plots_dir="plots"):
    import gmsh
    import meshio

    os.makedirs(outdir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    gmsh.initialize()
    gmsh.model.add("unit_square_with_circle")

    occ = gmsh.model.occ
    rect = occ.addRectangle(0.0, 0.0, 0.0, 1.0, 1.0)
    circ = occ.addDisk(cx, cy, 0.0, r, r)

    occ.fragment([(2, rect)], [(2, circ)])
    occ.synchronize()

    # Identify surfaces (2D) for physical groups via area
    surfs = [tag for (dim, tag) in gmsh.model.getEntities(dim=2)]
    areas = {s: gmsh.model.occ.getMass(2, s) for s in surfs}
    omega_tag = min(areas, key=areas.get)
    domain_tag = max(areas, key=areas.get)

    gmsh.model.addPhysicalGroup(2, [domain_tag], 1)
    gmsh.model.setPhysicalName(2, 1, "domain")
    gmsh.model.addPhysicalGroup(2, [omega_tag], 2)
    gmsh.model.setPhysicalName(2, 2, "omega")

    # Facets: classify circle vs outer boundary
    lines = [tag for (dim, tag) in gmsh.model.getEntities(dim=1)]
    circle_lines = []
    outer_lines = []
    for l in lines:
        xmin, ymin, _, xmax, ymax, _ = gmsh.model.getBoundingBox(1, l)
        xmid, ymid = 0.5 * (xmin + xmax), 0.5 * (ymin + ymax)
        if abs(np.hypot(xmid - cx, ymid - cy) - r) < 5e-2:
            circle_lines.append(l)
        else:
            outer_lines.append(l)
    if outer_lines:
        gmsh.model.addPhysicalGroup(1, outer_lines, 11)
        gmsh.model.setPhysicalName(1, 11, "outer")
    if circle_lines:
        gmsh.model.addPhysicalGroup(1, circle_lines, 12)
        gmsh.model.setPhysicalName(1, 12, "circle")

    # Global mesh size + optional refinement near circle
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", h)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", h)
    if refine_circle and circle_lines:
        field = gmsh.model.mesh.field
        fdist = field.add("Distance")
        field.setNumbers(fdist, "EdgesList", circle_lines)
        fth = field.add("Threshold")
        field.setNumber(fth, "InField", fdist)
        field.setNumber(fth, "SizeMin", 0.5 * h)
        field.setNumber(fth, "SizeMax", h)
        field.setNumber(fth, "DistMin", 0.25 * r)
        field.setNumber(fth, "DistMax", 0.75 * r)
        field.setAsBackgroundMesh(fth)

    gmsh.model.mesh.generate(2)

    msh_path = os.path.join(outdir, "mesh.msh")
    gmsh.write(msh_path)

    msh = meshio.read(msh_path)
    tri = msh.get_cells_type("triangle")
    tri_data = msh.get_cell_data("gmsh:physical", "triangle")
    line = msh.get_cells_type("line")
    line_data = msh.get_cell_data("gmsh:physical", "line") if len(line) else None

    # Write cell mesh (triangles) with subdomain tags
    cells = [("triangle", tri)]
    cell_data = {"subdomains": [tri_data]}
    meshio.write(
        os.path.join(outdir, "mesh_cells.xdmf"),
        meshio.Mesh(points=msh.points, cells=cells, cell_data=cell_data),
    )

    # Write facet mesh (lines) with facet tags
    if len(line):
        meshio.write(
            os.path.join(outdir, "mesh_facets.xdmf"),
            meshio.Mesh(
                points=msh.points,
                cells=[("line", line)],
                cell_data={"facets": [line_data]},
            ),
        )

    # Save a PNG of the triangular mesh (omega highlighted in red)
    try:
        _plot_mesh_png(
            points=msh.points,
            triangles=tri,
            outpath=os.path.join(plots_dir, "mesh.png"),
            h=h,
            tri_tags=tri_data,
            omega_id=2,
        )
    except Exception as e:
        print(f"Warning: could not save mesh PNG: {e}")

    gmsh.finalize()
    print(f"Written: {msh_path}\n  -> {outdir}/mesh_cells.xdmf\n  -> {outdir}/mesh_facets.xdmf")


def _plot_mesh_png(points, triangles, outpath: str, h: float | None = None, tri_tags=None, omega_id: int = 2):
    import matplotlib.tri as mtri
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    tri = mtri.Triangulation(points[:, 0], points[:, 1], triangles)
    plt.figure(figsize=(5, 5))
    if tri_tags is not None and len(tri_tags) == len(triangles):
        in_omega = (np.asarray(tri_tags) == int(omega_id)).astype(float)
        cmap = ListedColormap(["#f5f5f5", "#d62728"])
        plt.tripcolor(
            tri,
            facecolors=in_omega,
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            edgecolors='0.4',
            linewidth=0.45,
            alpha=1.0,
        )
        plt.legend(
            handles=[
                Patch(facecolor="#d62728", edgecolor='0.4', label=r'$\omega$'),
                Patch(facecolor="#f5f5f5", edgecolor='0.4', label=r'$\Omega\setminus\omega$'),
            ],
            loc='upper right',
            frameon=True,
            fontsize=9,
        )
    else:
        plt.triplot(tri, color='0.4', linewidth=0.5)
    if h is None:
        plt.title('Mesh of $[0,1]^2$ with $\\omega$')
    else:
        plt.title(rf'Mesh of $[0,1]^2$ with $\\omega$ ($h={h:g}$)')
    plt.xlabel('x'); plt.ylabel('y')
    plt.gca().set_aspect('equal', adjustable='box')
    plt.tight_layout()
    plt.savefig(outpath, dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cx", type=float, default=0.5)
    ap.add_argument("--cy", type=float, default=0.5)
    ap.add_argument("--r", type=float, default=0.1)
    ap.add_argument("--h", type=float, default=0.03)
    ap.add_argument("--no-refine-circle", action="store_true")
    ap.add_argument("--outdir", default="mesh/out")
    ap.add_argument("--plots-dir", default="plots")
    args = ap.parse_args()
    build_mesh(args.cx, args.cy, args.r, args.h, not args.no_refine_circle, args.outdir, args.plots_dir)

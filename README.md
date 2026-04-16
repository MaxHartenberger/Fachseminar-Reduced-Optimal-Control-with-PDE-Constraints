# Fachseminar — Reduced Optimal Control with PDE Constraints

This repository studies a linear-quadratic optimal control problem with an elliptic PDE on the unit square.  
Discretization is done with FEniCS (P1 FEM), meshes are generated with Gmsh, and optimization is evaluated with:

- Gradient Descent (fixed step $1/L$)
- Barzilai-Borwein (BB)
- Nesterov (constant $(m,L)$)
- Conjugate Gradient (CG) baseline for the SPD reduced system

## Student quick run (3 commands)

From repository root:

```bash
conda activate fenics
python Code/generate_meshes.py --h-list 0.06 0.04 0.03 0.02 0.015 0.012 0.01 0.008
python -m Code.run_optimizers --h-list 0.06 0.04 0.03 0.02 0.015 0.012 0.01 0.008 --beta 1e-3 --omega-id 2
```

Main artifacts are written to `results/` and `plots/`.

## Prerequisites

- A working Python environment with FEniCS/DOLFIN (for example a Conda env named `fenics`)
- Packages: `numpy`, `scipy`, `matplotlib`, `meshio`, `gmsh`, `fenics`/`dolfin`

## Repository Layout

- `Code/`
   - `gmsh_mesh.py` — create one mesh (Gmsh + XDMF conversion)
   - `generate_meshes.py` — create a mesh family for multiple `h`
   - `reduced_oc_model.py` — reduced PDE-constrained OC model and operators
   - `optimizers.py` — first-order optimization algorithms
   - `run_optimizers.py` — mesh-independence driver + plots + report tables
- `mesh/` — generated meshes (`mesh_h_*/mesh_cells.xdmf`, etc.)
- `plots/` — per-mesh and aggregated plots
- `results/` — per-mesh arrays, timing tables, and summary tables/JSON
- `report/` — LaTeX report
- `presentation/` — LaTeX Beamer slides

## Quickstart

### 1) Activate environment

```bash
conda activate fenics
```

### 2) Generate one mesh (optional)

```bash
python Code/gmsh_mesh.py --h 0.02 --outdir mesh/mesh_h_0.02 --plots-dir mesh/mesh_h_0.02
```

This writes:

- `mesh/mesh_h_0.02/mesh.msh`
- `mesh/mesh_h_0.02/mesh_cells.xdmf`
- `mesh/mesh_h_0.02/mesh_facets.xdmf`
- `mesh/mesh_h_0.02/mesh.png`

### 3) Generate a full mesh family (recommended)

```bash
python Code/generate_meshes.py --h-list 0.06 0.04 0.03 0.02 0.015 0.012 0.01 0.008
```

### 4) Run optimization study on existing meshes

```bash
python -m Code.run_optimizers \
   --h-list 0.06 0.04 0.03 0.02 0.015 0.012 0.01 0.008 \
   --beta 1e-3 \
   --omega-id 2
```

## Main Outputs

### Results (`results/`)

- `mesh_independence_summary.json`
- `mesh_independence_table.tex`
- `mesh_iterations_table.tex`
- Per mesh: `mesh_h_*/u_star.npy`, `mesh_h_*/y_star.npy`, `mesh_h_*/timings_table.tex`

### Plots (`plots/`)

- Aggregated:
   - `mesh_independence_cost_vs_h.png`
   - `mesh_independence_cost_vs_ndofs.png`
   - `mesh_independence_iterations_vs_h.png`
   - `mesh_independence_total_times_vs_h.png`
   - `mesh_independence_iteration_times_vs_h.png`
- Per mesh (optional): `plots/mesh_h_*/...`

## Useful CLI Options

### `Code/run_optimizers.py`

- `--h-list`: list of mesh sizes
- `--beta`: Tikhonov regularization parameter (default `1e-3`)
- `--omega-id`: physical cell tag of control region (default `2`)
- `--mesh-root`: root mesh directory (default `mesh`)
- `--mesh-prefix`: mesh directory prefix (default `mesh_h_`)
- `--plots-dir`: plot output directory (default `plots`)
- `--results-root`: results output directory (default `results`)
- `--no-per-mesh-plots`: disable per-mesh plot generation

### `Code/generate_meshes.py`

- `--h-list`: list of mesh sizes
- `--cx`, `--cy`, `--r`: circle center/radius for control region geometry
- `--no-refine-circle`: disable local refinement near the circle
- `--mesh-root`: mesh output root (default `mesh`)

## Build Report / Presentation

From repository root:

```bash
cd report
pdflatex -interaction=nonstopmode optimization_report.tex
pdflatex -interaction=nonstopmode optimization_report.tex

cd ../presentation
pdflatex -interaction=nonstopmode presentation.tex
pdflatex -interaction=nonstopmode presentation.tex
```

## Notes

- If an XDMF file is missing, run mesh generation first.
- The control region corresponds to physical cell tag `omega` (default ID `2`).
- MPI warnings from `dolfin` are often harmless in single-process runs.

# Fachseminar — PDE-Constrained Optimization (Ex3)

This repo solves a linear–quadratic optimal control problem with an elliptic PDE on the unit square, using FEniCS for discretization and first‑order methods (GD 1/L, Barzilai–Borwein, Nesterov) for optimization. Meshes come from Gmsh and are imported via XDMF.

## Prerequisites
- A working FEniCS environment (e.g., Conda env `fenics`).
- Python packages: `numpy`, `scipy`, `matplotlib`, `meshio`, `gmsh`, `fenics`/`dolfin`.

## Quickstart
1. Activate the environment:
   ```bash
   conda activate fenics
   ```
2. Generate the external mesh and a mesh plot:
   ```bash
   python mesh/gmsh_mesh.py --outdir mesh/out --plots-dir plots
   ```
   Outputs (under `mesh/out`): `mesh.msh`, `mesh_cells.xdmf`, `mesh_facets.xdmf`. Mesh plot saved to `plots/mesh.png`.
3. Run the optimization methods and save figures/results:
   ```bash
   python Code/run_methods.py --mesh-cells-xdmf mesh/out/mesh_cells.xdmf --omega-id 2 --beta 1e-3 --plots-dir plots --results-dir results
   ```

## Outputs
- Plots (in `plots/`):
  - `mesh.png`, `omega.png`, `y_d.png`, `u_opt.png`, `y_opt.png`, `grad_norm.png`, `cost.png`
- Results (in `results/`):
  - `optimizer_summary.json` (beta, omega-id, L estimate, final costs, best method, iterations)
  - `u_opt.npy`, `y_opt.npy`

## CLI Options (run_methods.py)
- `--mesh-cells-xdmf`: path to the XDMF cell mesh (default `mesh/out/mesh_cells.xdmf`).
- `--omega-id`: physical id of the control subdomain (default `2`).
- `--beta`: Tikhonov parameter (default `1e-3`).
- `--plots-dir`: where to save PNGs (default `plots`).
- `--results-dir`: where to save arrays/summary (default `results`).
- `--show`: display figures interactively.

## Notes
- The model assembles FE operators (A, M, M_U, B) and computes the reduced gradient via the adjoint equation. Stepsizes for GD use a Lipschitz estimate by power iteration.
- If you see `XDMF file ... does not exist`, run the mesh generation step first.
- MPI size warnings from `dolfin` can often be ignored in single‑process runs.

## Structure
- `mesh/`: mesh generation (`gmsh_mesh.py`) and XDMF outputs.
- `Code/`: scripts and modules (`run_methods.py`, `reduced_oc_model.py`, `optimizers.py`).
- `plots/`: generated figures.
- `results/`: numeric outputs and summary.

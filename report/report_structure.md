# Report — Comparison of Optimization Methods for a PDE-Constrained Optimal Control Problem

## Abstract


## Problem Statement

## Variational Formulation and Discretization

## Reduced Formulation


## Lipschitz Constant Estimation


## Optimization Methods
### Fixed-Step Gradient Descent (1/L)
- Update $u_{k+1}=u_k-(1/L)\nabla F(u_k)$; monotone decrease for quadratic convex cases.
- Diagnostics: plot $\|\nabla F(u_k)\|_U$ and $F(u_k)$; sensitivity to $L$ quality.
### Barzilai–Borwein (BB1/BB2)
- Secant-based step lengths in U-metric; alternating BB1/BB2, safeguards when $\langle s,d\rangle_U\approx 0$.
- Practical behavior: faster convergence; potential non-monotonicity.
### Nesterov’s Accelerated Gradient (with Restart)
- Momentum scheme with $1/L$ and restart on loss increase; acceleration and robustness trade-offs.


## Numerical Experiments
### Setup and Parameters
- Defaults: $\beta=10^{-3}$, $\omega$-id=2, mesh size $h$; $u_0=\nabla F(0)$; estimated $L$.
- Variations: sensitivity in $\beta$, mesh refinement $h$, restart settings.
### Figures and Convergence ($\|\nabla F\|_U$ and $F$)
- Include [plots/y_d.png](plots/y_d.png), [plots/u_opt.png](plots/u_opt.png), [plots/y_opt.png](plots/y_opt.png).
- Convergence: [plots/grad_norm.png](plots/grad_norm.png), [plots/cost.png](plots/cost.png); annotate best method from results.

## Discussion
- Method comparison (BB vs GD vs Nesterov): speed, stability, parameter sensitivity.
- Mesh-independence and effect of external $\omega$ tagging; accuracy near interface; limitations and improvements.

## Conclusion
- Key takeaways; recommended methods/parameters; outlook (constraints, higher-order elements, preconditioning).

## References
- Core texts on PDE-constrained optimization and first-order methods; lecture materials and articles.

## Appendix (Optional)
- Derivations (adjoint, gradient in U-metric), implementation notes (assembly, factorizations).
- Extra figures/parameter sweeps; environment and command references.
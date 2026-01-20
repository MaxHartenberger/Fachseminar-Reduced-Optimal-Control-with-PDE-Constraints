# Report — Comparison of Optimization Methods for a PDE-Constrained Optimal Control Problem

## Abstract
- Problem: PDE-constrained optimal control on $\Omega=(0,1)^2$ with circular $\omega$; objective $F(u)=\tfrac12\|y-y_d\|^2+\tfrac{\beta}{2}\|u\|^2$.
- Methods: GD with $1/L$, Barzilai–Borwein (BB1/BB2), Nesterov (restart); summary of results.

## Introduction
- Motivation and lecture context; prior work on gradient variants and BB strategies.
- Contributions and structure of the report; reproducibility via provided Python scripts and figures.

## Continuous Problem
- PDE: $-\Delta y=\chi_\omega u$ in $\Omega$, $y=0$ on $\partial\Omega$; target $y_d$; parameter $\beta>0$.
- Spaces and norms: $y\in H_0^1(\Omega)$, $u\in L^2(\Omega)$ (restricted to $\omega$ via $\chi_\omega$).

### Weak Formulation
- Variational form: find $y\in H_0^1$ with $\int_\Omega \nabla y\cdot\nabla v=\int_\omega u v$ for all $v\in H_0^1$.
- Optimality system: adjoint $p\in H_0^1$ via $\int_\Omega \nabla p\cdot\nabla w=\int_\Omega (y-y_d)w$; reduced-gradient condition.

## Finite Element Discretization
- Triangulation and $\mathbb{P}_1$ (CG1) elements; $V_h\subset H_0^1$, $U_h\subset L^2$.
- Dirichlet handling: apply BCs on $V_h$ only; $U_h$ shares basis without BCs. Include [plots/mesh.png](plots/mesh.png).
- Definitions: $A$ (stiffness on $\Omega$), $M$ (mass on $\Omega$), $M_U$ ($L^2$ mass on $U_h$), $B$ (control coupling on $\omega$).
- Assembly via XDMF subdomain tags; apply Dirichlet to $A$ only; zero RHS at boundary DOFs for state/adjoint.
- Inner product: $(a,b)_U=a^T M_U b$, norm $\|a\|_U$.
- Gradient: $\nabla F(u)=M_U^{-1} B^T p + \beta u$, with adjoint $A p = M(y-y_d)$; Riesz Hessian $H=M_U^{-1}(B^T A^{-1} M A^{-1} B + \beta M_U)$.

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
# Interview Q&A — Vehicle Aerodynamic Shape Optimizer — Bayesian Optimization
## Gaussian Process + Expected Improvement Acquisition | Aerodynamic Design Optimization

### Q1: What is Bayesian Optimization?

Bayesian Optimization is a global optimization framework for expensive black-box functions. It builds a probabilistic surrogate (GPR) of the objective function, then uses an acquisition function (e.g., Expected Improvement, Upper Confidence Bound) to balance exploitation (evaluating near the current best) and exploration (evaluating high-uncertainty regions). It is sample-efficient — finding good optima in far fewer evaluations than random search or grid search.

---

### Q2: What is Expected Improvement (EI)?

EI(x) = E[max(f(x*) − f(x), 0)] where f(x*) is the current best observed value. It combines the GPR mean (exploitation) and standard deviation (exploration) into a single criterion: high EI where the surrogate predicts improvement AND where it is uncertain. Integrating over the GPR distribution yields a closed-form expression involving the normal CDF and PDF.

---

### Q3: What is Latin Hypercube Sampling (LHS)?

LHS is a stratified sampling method that divides each design variable's range into N equal intervals and places exactly one sample in each interval for each variable, while randomizing the combinations. This ensures uniform coverage of the design space — much better than random sampling, which tends to cluster samples. LHS is the standard DoE method for surrogate-based optimization in aerospace and automotive.

---

### Q4: What is the difference between local and global optimization?

Local optimization algorithms (gradient descent, L-BFGS-B) find the nearest local minimum from a starting point. Global optimization (Bayesian Optimization, genetic algorithms, simulated annealing) searches the entire design space for the global minimum, even in non-convex landscapes with multiple local optima. Aerodynamic design spaces often have complex multi-modal landscapes requiring global methods.

---

### Q5: How does this compare to adjoint-based aerodynamic optimization?

Adjoint methods compute the gradient of Cd with respect to every surface point efficiently, enabling gradient-based shape optimization directly in the CFD solver (used in OpenFOAM adjoint, ANSYS Fluent adjoint, and all F1 CFD workflows). Adjoint requires access to the solver internals. Bayesian Optimization is solver-agnostic (works with any simulation tool) but is limited to O(100) evaluations. They are complementary: BO for coarse exploration, adjoint for local refinement.

---


# Technical Notes — Aerodynamic Bayesian Optimization

## Purpose

This repository demonstrates sequential optimization of a synthetic aerodynamic objective using a Gaussian Process surrogate and Expected Improvement.

## Workflow

1. Generate an initial design sample.
2. Fit a Gaussian Process surrogate.
3. Evaluate Expected Improvement over candidate locations.
4. Select the next design to evaluate.
5. Update the surrogate and repeat.

## Objective Scope

The drag objective is synthetic / physics-inspired. It is used to demonstrate the optimization mechanism without requiring CFD during each iteration.

## Interpretation

The reported objective reduction is valid only for the generated design landscape.

The technical value of the project is the search strategy:
- predictive mean;
- predictive uncertainty;
- acquisition function;
- exploration/exploitation trade-off;
- convergence under a limited evaluation budget.

## Limitations

A production aerodynamic optimizer would require:
- CFD or experimental objective evaluations;
- geometric feasibility constraints;
- mesh and solver convergence controls;
- multi-objective trade-offs;
- validation of the final design with high-fidelity simulation.

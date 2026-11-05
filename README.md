# Vehicle Aerodynamic Shape Optimization — Bayesian Optimization

> Gaussian-process surrogate modelling with Expected Improvement for aerodynamic design search

## Overview

This project demonstrates sequential engineering optimization when each objective evaluation is assumed to be expensive.

A Gaussian Process surrogate is fitted to a synthetic aerodynamic design landscape. Expected Improvement is then used to select new candidate designs and progressively search for lower drag coefficient.

## Method

Core components:
- Latin Hypercube / initial design sampling;
- Gaussian Process regression;
- uncertainty-aware surrogate prediction;
- Expected Improvement acquisition;
- sequential candidate selection;
- convergence and design-space visualization.

## Data Scope

The aerodynamic objective is **synthetic / physics-inspired**. The project demonstrates Bayesian-optimization mechanics and engineering interpretation; it is not a CFD-validated aerodynamic optimization study.

## Reported Result

The saved experiment reports a reduction in the synthetic drag objective from approximately:

```text
Cd = 0.30 → 0.153
```

within the configured evaluation budget.

That result applies only to the repository's generated objective landscape.

## Run

```bash
git clone https://github.com/bharath153/Aero_BayesOpt.git
cd Aero_BayesOpt
pip install -r requirements.txt
python src/train_bayesopt.py
```

## Repository Structure

```text
Aero_BayesOpt/
├── src/
│   └── train_bayesopt.py
├── plots/
├── docs/
├── metrics.json
├── requirements.txt
└── README.md
```

## Relationship to the GPR Surrogate Project

[Aero_Drag_GPR](https://github.com/bharath153/Aero_Drag_GPR) focuses on surrogate prediction. This repository extends the same general idea to **sequential decision-making**, where predictive uncertainty is used to decide where to evaluate next.

## Possible Extensions

- CFD-backed objective evaluations;
- multiple aerodynamic objectives;
- geometric constraints;
- batch Bayesian optimization;
- multi-fidelity surrogates;
- comparison with evolutionary algorithms.

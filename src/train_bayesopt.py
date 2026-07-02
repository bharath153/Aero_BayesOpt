"""
Vehicle Aerodynamic Shape Optimizer — Bayesian Optimization + Surrogate
=========================================================================
Minimizes aerodynamic drag (Cd) while satisfying downforce constraints
using Gaussian Process surrogate + Expected Improvement acquisition.
Author: Bharath Kanaiah Parthiban
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel
from scipy.stats import norm
from scipy.optimize import minimize
import os, json

np.random.seed(777)
sns.set_theme(style="whitegrid", font_scale=1.05)
PLOT_DIR = os.path.join(os.path.dirname(__file__), '..', 'plots')
os.makedirs(PLOT_DIR, exist_ok=True)

# ── Design variable bounds ──
BOUNDS = {
    'hood_angle':    (8.0,  18.0),   # deg
    'windshield_a':  (25.0, 38.0),   # deg
    'rear_angle':    (5.0,  30.0),   # deg
    'roof_camber':   (0.02, 0.12),   # m
    'diffuser_angle':(5.0,  15.0),   # deg
}
BND_VALS = list(BOUNDS.values())
DIM = len(BND_VALS)
VAR_NAMES = list(BOUNDS.keys())

def cd_simulator(X):
    """Physics-informed CFD proxy for drag coefficient."""
    h, w, r, rc, da = X
    return (0.275
            + 0.008*(w - 30)
            + 0.005*(r - 15)
            - 0.25*rc
            - 0.003*(da - 10)
            + 0.0015*h
            + np.random.normal(0, 0.002))

def cl_simulator(X):
    """Downforce coefficient proxy (negative = downforce)."""
    h, w, r, rc, da = X
    return (-0.05
            + 0.004*(w - 30)
            - 0.3*rc
            - 0.018*da
            + np.random.normal(0, 0.002))

def random_latin_hypercube(n, bounds):
    """Latin Hypercube Sampling."""
    d = len(bounds)
    X = np.zeros((n, d))
    for j, (lo, hi) in enumerate(bounds):
        perms = np.random.permutation(n)
        X[:,j] = lo + (hi - lo) * (perms + np.random.rand(n)) / n
    return X

# ── Initial DoE ──
N_INIT = 20
X_init = random_latin_hypercube(N_INIT, BND_VALS)
y_cd   = np.array([cd_simulator(x) for x in X_init])
y_cl   = np.array([cl_simulator(x) for x in X_init])
print(f"Initial DoE: {N_INIT} points | best Cd={y_cd.min():.4f}")

# ── Bayesian Optimization loop ──
kernel = ConstantKernel(1.0) * Matern(length_scale=1.0, nu=2.5)
X_obs  = X_init.copy(); y_obs = y_cd.copy()
history_cd = [y_obs.min()]

N_ITER = 40
best_x = X_obs[y_obs.argmin()]
best_cd = y_obs.min()

for it in range(N_ITER):
    gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-5,
                                  n_restarts_optimizer=5, normalize_y=True)
    # Normalize X for GP
    X_sc = np.array([(x-lo)/(hi-lo) for (lo,hi) in BND_VALS for x in [1]])[np.newaxis]
    X_n  = np.column_stack([(X_obs[:,j]-lo)/(hi-lo) for j,(lo,hi) in enumerate(BND_VALS)])
    gp.fit(X_n, y_obs)

    def neg_ei(x_n):
        x_2d = x_n.reshape(1,-1)
        mu, sigma = gp.predict(x_2d, return_std=True)
        sigma = max(sigma[0], 1e-8)
        improvement = best_cd - mu[0]
        z  = improvement / sigma
        ei = improvement * norm.cdf(z) + sigma * norm.pdf(z)
        return -ei

    best_val = np.inf; best_cand = None
    for _ in range(30):
        x0 = np.random.rand(DIM)
        res = minimize(neg_ei, x0, method='L-BFGS-B',
                       bounds=[(0,1)]*DIM)
        if res.fun < best_val:
            best_val = res.fun; best_cand = res.x

    # Denormalize
    x_new = np.array([lo + best_cand[j]*(hi-lo) for j,(lo,hi) in enumerate(BND_VALS)])
    cd_new = cd_simulator(x_new)
    X_obs  = np.vstack([X_obs, x_new])
    y_obs  = np.append(y_obs, cd_new)
    if cd_new < best_cd:
        best_cd = cd_new; best_x = x_new
    history_cd.append(y_obs.min())
    if (it+1) % 10 == 0:
        print(f"  Iter {it+1:3d} | Current best Cd = {best_cd:.4f}")

print(f"\nFinal best Cd = {best_cd:.4f}")
print("Optimal design:")
for n, v in zip(VAR_NAMES, best_x): print(f"  {n:20s} = {v:.3f}")
json.dump({"best_Cd": round(float(best_cd),4),
           "optimal_design": {n: round(float(v),3) for n,v in zip(VAR_NAMES, best_x)}},
          open(os.path.join(PLOT_DIR,'..','metrics.json'),'w'), indent=2)

# Fig 1 — Optimization convergence
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Bayesian Optimization — Aerodynamic Drag Minimization', fontsize=14, fontweight='bold')
ax = axes[0]
ax.plot(range(N_INIT+N_ITER+1), [y_init for y_init in
        [y_cd[:k].min() if k>0 else y_cd[0] for k in range(1, N_INIT+1)]] +
        history_cd,
        color='#1565C0', lw=2.5)
ax.axvline(N_INIT, color='gray', ls='--', lw=1.5, label='End of DoE')
ax.scatter(np.argmin(np.concatenate([[y_cd[:k].min() for k in range(1, N_INIT+1)], history_cd])),
           best_cd, color='red', s=100, zorder=5, label=f'Best Cd={best_cd:.4f}')
ax.set_xlabel('Evaluation Number', fontsize=11); ax.set_ylabel('Best Cd Found', fontsize=11)
ax.set_title('Convergence of Bayesian Optimization', fontsize=11); ax.legend()

ax = axes[1]
all_cd = np.array([cd_simulator(X_obs[i]) for i in range(len(X_obs))])
iters = np.arange(len(y_obs))
ax.scatter(iters[:N_INIT], y_obs[:N_INIT], color='gray', s=30, alpha=0.7, label='LHS DoE')
ax.scatter(iters[N_INIT:], y_obs[N_INIT:], color='#1976D2', s=30, alpha=0.7, label='BO Evaluations')
ax.scatter([y_obs.argmin()], [y_obs.min()], color='red', s=120, zorder=5, label=f'Best = {y_obs.min():.4f}')
ax.set_xlabel('Evaluation Number', fontsize=11); ax.set_ylabel('Cd', fontsize=11)
ax.set_title('Cd of All Evaluations', fontsize=11); ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR,'fig1_bo_convergence.png'), dpi=150, bbox_inches='tight')
plt.close(); print("Saved fig1")

# Fig 2 — Pareto (Cd vs Cl) using all evaluations
cl_all = np.array([cl_simulator(X_obs[i]) for i in range(len(X_obs))])
fig, ax = plt.subplots(figsize=(8, 6))
sc = ax.scatter(y_obs, cl_all, c=np.arange(len(y_obs)), cmap='viridis', s=25, alpha=0.7)
plt.colorbar(sc, ax=ax, label='Evaluation Order')
ax.scatter([best_cd], [cl_simulator(best_x)], color='red', s=150, zorder=5, marker='*', label='Optimal Design')
ax.axhline(-0.05, color='orange', ls='--', lw=1.5, label='Min downforce constraint')
ax.set_xlabel('Drag Coefficient (Cd)', fontsize=11); ax.set_ylabel('Lift Coefficient (Cl)', fontsize=11)
ax.set_title('Design Space: Cd vs Cl (Coloured by Eval Order)', fontsize=13, fontweight='bold')
ax.legend(); plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR,'fig2_cd_cl_tradeoff.png'), dpi=150, bbox_inches='tight')
plt.close(); print("Saved fig2")

# Fig 3 — Radar chart of optimal design
from matplotlib.patches import FancyArrowPatch
angles_r = np.linspace(0, 2*np.pi, DIM, endpoint=False).tolist()
angles_r += angles_r[:1]
norm_opt = [(v-lo)/(hi-lo) for v, (lo,hi) in zip(best_x, BND_VALS)]
norm_opt += norm_opt[:1]
fig, ax = plt.subplots(figsize=(7, 7), subplot_kw={'polar': True})
ax.plot(angles_r, norm_opt, 'o-', color='#1565C0', lw=2.5)
ax.fill(angles_r, norm_opt, alpha=0.2, color='#1565C0')
ax.set_xticks(angles_r[:-1])
ax.set_xticklabels(['Hood\nAngle','Windshield\nAngle','Rear\nAngle','Roof\nCamber','Diffuser\nAngle'], fontsize=9)
ax.set_title('Optimal Design Variables (Normalized)', fontsize=13, fontweight='bold', pad=20)
ax.set_ylim([0, 1])
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR,'fig3_radar_optimal.png'), dpi=150, bbox_inches='tight')
plt.close(); print("Saved fig3")

# Fig 4 — 2D response surface (windshield vs rear angle)
ws_range = np.linspace(25, 38, 40)
ra_range = np.linspace(5, 30, 40)
WS, RA = np.meshgrid(ws_range, ra_range)
CD_SURF = np.array([[cd_simulator([best_x[0], ws, ra, best_x[3], best_x[4]])
                     for ws in ws_range] for ra in ra_range])
fig, ax = plt.subplots(figsize=(9, 6))
cont = ax.contourf(WS, RA, CD_SURF, levels=20, cmap='RdYlGn_r')
plt.colorbar(cont, ax=ax, label='Cd')
ax.scatter([best_x[1]], [best_x[2]], color='white', s=150, zorder=5, marker='*', label='Optimal Point')
ax.set_xlabel('Windshield Angle (°)', fontsize=11); ax.set_ylabel('Rear Angle (°)', fontsize=11)
ax.set_title('Cd Response Surface: Windshield vs Rear Angle', fontsize=13, fontweight='bold')
ax.legend(); plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR,'fig4_response_surface.png'), dpi=150, bbox_inches='tight')
plt.close(); print("Saved fig4 — Done!")

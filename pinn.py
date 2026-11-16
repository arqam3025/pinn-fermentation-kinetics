"""Project 1 - Physics-informed neural network for fermentation kinetics.

Goal: from a handful of noisy biomass/substrate samples taken during the
*first part* of a batch fermentation, (a) recover the Monod kinetic
parameters (mu_max, Ks, Y) and (b) forecast the rest of the run.

Batch Monod model:
    dX/dt =  mu(S) X
    dS/dt = -mu(S) X / Y,      mu(S) = mu_max S / (Ks + S)

Three approaches are compared on the same data:
    1. Pure neural network fitted to the data only (no physics)
    2. PINN: network + ODE residual loss, with the kinetic parameters learned
    3. Classical reference: ODE least-squares fit (scipy)

    python p1_pinn_kinetics/pinn.py            # ~2-4 min on CPU
"""
from __future__ import annotations

import json
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
torch.set_num_threads(1)                # tiny MLPs: extra threads only add overhead
from scipy.integrate import solve_ivp  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402

OUT = pathlib.Path(__file__).parent / "results"
T_END = 40.0                       # h
X0, S0 = 0.5, 20.0                 # g/L
X_SCALE, S_SCALE = 12.0, 20.0      # output scaling for the networks


def monod_rhs(t, y, mu_max, Ks, Y):
    X, S = y
    S = max(S, 0.0)
    mu = mu_max * S / (Ks + S)
    return [mu * X, -mu * X / Y]


def simulate(mu_max, Ks, Y, t):
    sol = solve_ivp(monod_rhs, (0, T_END), [X0, S0], t_eval=t, args=(mu_max, Ks, Y),
                    rtol=1e-8, atol=1e-10)
    return sol.y.T                  # (len(t), 2)


class Net(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.f = nn.Sequential(nn.Linear(1, hidden), nn.Tanh(), nn.Linear(hidden, hidden), nn.Tanh(),
                               nn.Linear(hidden, hidden), nn.Tanh(), nn.Linear(hidden, 2))

    def forward(self, t):                      # t in hours
        z = self.f(t / T_END)
        X = X_SCALE * nn.functional.softplus(z[:, :1])
        S = S_SCALE * nn.functional.softplus(z[:, 1:])
        return X, S


def train_nn(t_obs, y_obs, physics: bool, seed: int, steps: int = 6000,
             warmup: int = 1500, ramp: int = 1500, lam: float = 0.1):
    torch.manual_seed(seed)
    net = Net()
    # kinetic parameters in log space; deliberately poor initial guesses
    log_mu = nn.Parameter(torch.tensor(np.log(0.5), dtype=torch.float32))
    log_ks = nn.Parameter(torch.tensor(np.log(1.0), dtype=torch.float32))
    log_y = nn.Parameter(torch.tensor(np.log(0.3), dtype=torch.float32))
    params = list(net.parameters()) + ([log_mu, log_ks, log_y] if physics else [])
    opt = torch.optim.Adam(params, lr=3e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)

    t_d = torch.tensor(t_obs, dtype=torch.float32)[:, None]
    X_d = torch.tensor(y_obs[:, 0], dtype=torch.float32)[:, None]
    S_d = torch.tensor(y_obs[:, 1], dtype=torch.float32)[:, None]
    t_col = torch.linspace(0, T_END, 400)[:, None].requires_grad_(True)   # collocation pts

    # Curriculum: fit the data first, then ramp in the ODE residual.  Without
    # this the physics term dominates early and the network collapses to the
    # trivial solution X(t) = 0, which satisfies dX/dt = mu*X for any mu.
    for step in range(steps):
        opt.zero_grad()
        Xp, Sp = net(t_d)
        loss = ((Xp - X_d) / X_SCALE).pow(2).mean() + ((Sp - S_d) / S_SCALE).pow(2).mean()
        w_phys = lam * min(1.0, max(0.0, (step - warmup) / ramp))
        if physics and w_phys > 0:
            X, S = net(t_col)
            dX = torch.autograd.grad(X.sum(), t_col, create_graph=True)[0]
            dS = torch.autograd.grad(S.sum(), t_col, create_graph=True)[0]
            mu = log_mu.exp() * S / (log_ks.exp() + S)
            r1 = (dX - mu * X) / X_SCALE * T_END
            r2 = (dS + mu * X / log_y.exp()) / S_SCALE * T_END
            X_0, S_0 = net(torch.zeros(1, 1))
            ic = ((X_0 - X0) / X_SCALE).pow(2).sum() + ((S_0 - S0) / S_SCALE).pow(2).sum()
            loss = loss + w_phys * (r1.pow(2).mean() + r2.pow(2).mean() + ic)
        loss.backward()
        opt.step()
        sched.step()
    est = dict(mu_max=log_mu.exp().item(), Ks=log_ks.exp().item(), Y=log_y.exp().item()) if physics else None
    return net, est


def classical_fit(t_obs, y_obs):
    def resid(p):
        mu, ks, yy = np.exp(p)
        pred = simulate(mu, ks, yy, t_obs)
        return np.concatenate([(pred[:, 0] - y_obs[:, 0]) / X_SCALE, (pred[:, 1] - y_obs[:, 1]) / S_SCALE])
    r = least_squares(resid, np.log([0.5, 1.0, 0.3]), bounds=(np.log([0.01, 1e-3, 0.05]), np.log([2, 10, 1.5])))
    mu, ks, yy = np.exp(r.x)
    return dict(mu_max=mu, Ks=ks, Y=yy)


def run_case(true, noise, seed, obs_until=0.4, n_obs=10):
    rng = np.random.default_rng(seed)
    t_obs = np.sort(rng.uniform(1.0, obs_until * T_END, n_obs))
    y_true_obs = simulate(**true, t=t_obs)
    y_obs = y_true_obs * (1 + noise * rng.normal(size=y_true_obs.shape))
    y_obs = np.clip(y_obs, 0, None)

    t_grid = np.linspace(0, T_END, 200)
    y_grid = simulate(**true, t=t_grid)
    future = t_grid > obs_until * T_END

    nn_plain, _ = train_nn(t_obs, y_obs, physics=False, seed=seed)
    pinn, est = train_nn(t_obs, y_obs, physics=True, seed=seed)
    cls = classical_fit(t_obs, y_obs)

    tg = torch.tensor(t_grid, dtype=torch.float32)[:, None]
    with torch.no_grad():
        preds = {name: np.hstack([m(tg)[0].numpy(), m(tg)[1].numpy()]) for name, m in
                 [("Neural net (data only)", nn_plain), ("PINN", pinn)]}
    preds["ODE least squares"] = simulate(**cls, t=t_grid)

    def rmse(p):   # forecast error on the unobserved part of the run (biomass, g/L)
        return float(np.sqrt(np.mean((p[future, 0] - y_grid[future, 0]) ** 2)))

    return dict(t_obs=t_obs, y_obs=y_obs, t_grid=t_grid, y_grid=y_grid, preds=preds,
                pinn_params=est, ls_params=cls,
                forecast_rmse={k: rmse(v) for k, v in preds.items()})


def main():
    OUT.mkdir(exist_ok=True)
    true = dict(mu_max=0.15, Ks=0.6, Y=0.55)
    summary = {"true_params": true, "cases": []}

    for noise in [0.02, 0.05, 0.10]:
        for seed in range(3):
            r = run_case(true, noise, seed)
            summary["cases"].append(dict(noise=noise, seed=seed, pinn_params=r["pinn_params"],
                                         ls_params=r["ls_params"], forecast_rmse=r["forecast_rmse"]))
            print(f"noise={noise:.2f} seed={seed} | PINN {r['pinn_params']} | RMSE {r['forecast_rmse']}", flush=True)
            if noise == 0.05 and seed == 0:
                showcase = r

    # --- figure: one showcase run ------------------------------------------
    r = showcase
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    styles = {"Neural net (data only)": ("tab:red", "--"), "PINN": ("tab:blue", "-"),
              "ODE least squares": ("tab:green", ":")}
    for j, (lab, unit) in enumerate([("Biomass X", "g/L"), ("Substrate S", "g/L")]):
        ax[j].plot(r["t_grid"], r["y_grid"][:, j], color="black", lw=2, alpha=0.35, label="True")
        for name, p in r["preds"].items():
            c, ls = styles[name]
            ax[j].plot(r["t_grid"], p[:, j], color=c, ls=ls, label=name)
        ax[j].scatter(r["t_obs"], r["y_obs"][:, j], color="black", zorder=5, s=18, label="Noisy samples")
        ax[j].axvspan(0.4 * T_END, T_END, color="grey", alpha=0.1)
        ax[j].set_title(f"{lab} (shaded = forecast, no data)")
        ax[j].set_xlabel("time [h]"); ax[j].set_ylabel(unit)
    ax[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "pinn_forecast.png", dpi=150)

    # --- aggregate table ---------------------------------------------------
    agg = {}
    for noise in [0.02, 0.05, 0.10]:
        cs = [c for c in summary["cases"] if c["noise"] == noise]
        agg[str(noise)] = {
            "forecast_rmse_mean": {k: float(np.mean([c["forecast_rmse"][k] for c in cs])) for k in cs[0]["forecast_rmse"]},
            "pinn_param_rel_error_pct": {k: float(np.mean([abs(c["pinn_params"][k] - true[k]) / true[k] * 100 for c in cs])) for k in true},
            "ls_param_rel_error_pct": {k: float(np.mean([abs(c["ls_params"][k] - true[k]) / true[k] * 100 for c in cs])) for k in true},
        }
    summary["aggregate"] = agg
    (OUT / "metrics.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(agg, indent=2))


if __name__ == "__main__":
    main()

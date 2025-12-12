#!/usr/bin/env python
# coding: utf-8

# In[26]:


#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import reservoirpy as rpy
from reservoirpy.nodes import Reservoir, Ridge
from bayes_opt import BayesianOptimization
import logging
import warnings

#Silence Libararies
#------------------
warnings.filterwarnings("ignore")
logging.getLogger("reservoirpy").setLevel(logging.CRITICAL)
logging.getLogger("matplotlib").setLevel(logging.ERROR)

# Reproducibility
rpy.set_seed(42)
np.random.seed(42)

# LORENZ-63 SYSTEM
#-----------------
def lorenz(state, t, sigma=10.0, rho=28.0, beta=8/3):
    """Lorenz63 system (standard parameters)"""
    x, y, z = state
    return [sigma * (y - x),
            x * (rho - z) - y,
            x * y - beta * z]

# Integrator Method
def generate_lorenz_data(rho=28.0, t_max=300.0, dt=0.01, burn_in=3000):
    """Generate Lorenz trajectory with random initial condition."""
    t = np.arange(0, t_max, dt)
    x0 = [-8.0, 8.0, 27.0] + 3.0 * np.random.randn(3)
    sol = odeint(lorenz, x0, t, args=(10.0, rho, 8/3))
    return t[burn_in:], sol[burn_in:]

# TRAINING DATA (ρ = 28.0)
#----------------------------
t_train, data_train = generate_lorenz_data(rho=28.0, t_max=200.0, burn_in=2000)
X_train_raw = data_train[:-1]
Y_train_raw = data_train[1:]

# Normalization
mean_X = X_train_raw.mean(axis=0)
std_X  = X_train_raw.std(axis=0)
std_X[std_X == 0] = 1.0

X_train = (X_train_raw - mean_X) / std_X
Y_train = (Y_train_raw - mean_X) / std_X

# RESERVOIR MODEL
#----------------------------------------------------------
neurons = 400

reservoir = Reservoir(neurons,
                      sr=0.8,
                      lr=0.390,
                      input_scaling=0.20,
                      input_connectivity=0.1,
                      rc_connectivity=0.30,
                      seed=42)

readout = Ridge(ridge=1e-5)
esn = reservoir >> readout

esn.fit(X_train, Y_train, warmup=200)

# FORECASTING
#------------
def forecast_synced(start_idx, test_data_raw, steps=100, warmup_steps=500):
    
    # Reset & warm-up
    reservoir.state_ = np.zeros((1, neurons))  # clean state (batch dim = 1)

    warmup_start = max(0, start_idx - warmup_steps)
    warmup_seq   = test_data_raw[warmup_start:start_idx]

    # Normalize
    if len(warmup_seq) > 0:
        warmup_norm = (warmup_seq - mean_X) / std_X
        reservoir.run(warmup_norm)                    # batch synchronisation (fast)

    # Autonomous generation
    pred_n = []
    current_n = (test_data_raw[start_idx] - mean_X) / std_X  
    pred_n.append(current_n.copy())

    for _ in range(steps):
        next_n = esn(current_n).flatten()                 
        pred_n.append(next_n.copy())
        current_n = next_n

    pred_n = np.array(pred_n)
    return pred_n * std_X + mean_X        # denormalise

# EVALUATION ON ANY ρ
#-----------------------
def evaluate_on_rho(rho, steps=30, n_trials=200, warmup_steps=500,
                    t_max=400.0, burn_in=4000):
    """Mean +- std MSE over many random starting points at given ρ."""
    _, test_data_raw = generate_lorenz_data(rho=rho, t_max=t_max, burn_in=burn_in)

    rng = np.random.default_rng(42)
    max_start = len(test_data_raw) - steps - warmup_steps - 20   # safety margin
    indices = rng.integers(0, max_start, size=n_trials)

    mses = []
    for idx in indices:
        pred = forecast_synced(idx, test_data_raw, steps=steps, warmup_steps=warmup_steps)
        true = test_data_raw[idx:idx + steps + 1]
        mses.append(np.mean((pred - true)**2))

    mses = np.array(mses)
    return mses.mean(), mses.std()

# SENSITIVITY ANALYSIS
#---------------------
rhos = np.linspace(22, 34, 7)
results = []

print("ρ      | 100-step MSE (± std)")
print("-" * 34)
for rho in rhos:
    mean_mse, std_mse = evaluate_on_rho(rho, steps=100, n_trials=200, warmup_steps=500)
    results.append((rho, mean_mse, std_mse))
    print(f"{rho:5.1f}  |  {mean_mse:.2e} ± {std_mse:.2e}")

# Plot
rhos_arr, mses, stds = zip(*results)
plt.figure(figsize=(8.5, 5))
plt.errorbar(rhos_arr, mses, yerr=stds, capsize=4, marker='o', linewidth=2, color='tab:blue')
plt.yscale('log')
plt.xlabel("Rayleigh number ρ", fontsize=12)
plt.ylabel("100-step ahead MSE (200 trials)", fontsize=12)
plt.title("Echo State Network Robustness to ρ\n(trained only on ρ = 28.0)", fontsize=14, weight='bold')
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()


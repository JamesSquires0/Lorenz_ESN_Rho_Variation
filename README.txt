README

This script trains an Echo State Network (ESN) on the Lorenz-63 system at ρ = 28 and evaluates how forecasting error changes when the Rayleigh number ρ is varied. 

The code:

Generates Lorenz trajectories using scipy.integrate.odeint

Normalizes training data

Trains an ESN (400 nodes, spectral radius 0.8, leakage rate 0.390)

Uses warmup steps to synchronize reservoir state

Tests forecasting performance across ρ \in {22, 24, 26, 28, 30, 32, 34}

Computes 100-step MSE over 200 trials per ρ

Plots MSE vs ρ on a log scale

To run:

python James_Squires_FinalProject_Code.py


Dependencies:

numpy
scipy
matplotlib
reservoirpy
bayesian-optimization


The script is fully reproducible with fixed random seeds.

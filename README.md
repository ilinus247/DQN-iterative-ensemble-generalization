# Code setup for experiments on RL generalizability through competitive MARL using PyTorch/TorchRL for algorithm implementation, Tensorboard for logging and metric visualization, and imageio for policy visualization.

The core idea behind experiments in this repository is to use iterative counter-adaptation of competitive MARL policies to create an ensemble of diverse and self-improved policies to improve the generalization of policies against environmental or adversarial shifts. The simple intuition behind this is that in the case of competitive MARL, both policies are seeking to exploit faults in the other. If we allow one side to 'overfit' to these faults, then the other can identify and adapt against the exploit when trained later. Collecting such policies, then, should produce a portfolio of them resistant to various exploits.  

An algorithm like...
*    1. Freeze Ad, train ONLY Ag for K iterations  -> Ag'
*    2. Freeze Ag', train ONLY Ad for K iterations -> Ad'
*    3. Save Ad' as ensemble member
* can produce one ensemble member per iteration, so that every Ad' is adapted to the faults of Ag', that was in turn adapted to the faults of Ad.

## Current Progress

Naive implementation of the algorithm with a uniformly-random member selection based ensemble directly overfits compared to the seed policy.

Metrics produced by evaluate_ensemble_generalization.py:
* Ensemble vs. single, same condition, side by side:
*   \[original agent / normal] single=105.40  ensemble=213.40  delta=+108.00
*   \[original agent / respawn_at_catch] single=30.80  ensemble=43.70  delta=+12.90
*   \[random agent / normal] single=456.60  ensemble=382.80  delta=-73.80
*   \[random agent / respawn_at_catch] single=81.10  ensemble=70.10  delta=-11.00

This is expected considering every Ad' is a direct improvement of its Ad. Techniques to combat this overfitting will be needed. Next steps are employing divergence metrics on the ensemble policies to confirm they are diverse as opposed to simply being better versions of the previous, investigating underfitting techniques and ensuring their effect is accounted for in comparison evaluations against the original policy.  

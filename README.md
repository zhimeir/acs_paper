# acs_paper
This repository contains the code for reproducing the numerical results in [ACS: An interactive framework for conformal selection](https://arxiv.org/pdf/2507.15825).

## Simulations
The code for replicating the simulation results is in the `simulation` folder, and has been tested in Python 3.11.4. 
The scripts are written to automatically carry out experiments with different configurations and random seeds. 
- To carry out one run (task id: 1) of the experiments with base learners (Section 5.2), run the following code in the terminal:
```
python3 base_learner.py 1
```


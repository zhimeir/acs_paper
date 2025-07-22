# acs_paper
This repository contains the code for reproducing the numerical results in [ACS: An interactive framework for conformal selection](https://arxiv.org/pdf/2507.15825).

## Simulations
The code for replicating the simulation results is in the `simulation` folder, and has been tested in Python 3.11.4. 
The scripts are written to carry out experiments with different configurations and random seeds, indexed by `task_id`.
For example, `task_id=1` refers to the experiment under Setting 1, noise level 0.3, seed group 1, base learner SVR, calibration size 200.  
- To carry out one run (with `task id=1`) of the experiments with base learners (Section 5.2), run the following code in the terminal under `/ACS_paper`:
```
python3 base_learner.py 1
```
- To carry out one run (with `task id=1`) of the experiments with adaptive model selection (Section 5.3), run the following code in the terminal under `/ACS_paper`:
```
python3 model_selection.py 1
```
- To carry out one run (with `task id=1`) of the experiments with extra labels (Section 5.4), run the following code in the terminal under `/ACS_paper`:
```
python3 extra_label.py 1
```
- To carry out one run (with `task id=1`) of the experiments with diversity-aware selection (Section 5.5), run the following code in the terminal under `/ACS_paper`:
```
python3 diversity.py 1
```
The results can be found in the `results` folder.


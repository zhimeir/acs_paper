import numpy as np
import pandas as pd
import argparse
import os
import subprocess
import sys
git_root = subprocess.check_output(['git', 'rev-parse', '--show-toplevel']).strip().decode('utf-8')
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
from sklearn.model_selection import ParameterGrid
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVR
from utils.cs_utils import gen_data, BH
from utils.acs import *

from sklearn.metrics import accuracy_score

""" Configurations of the current run """
parser = argparse.ArgumentParser('')
parser.add_argument('--task_id', type=int, default=1)
args = parser.parse_args()
task_id = args.task_id - 1 
params = {'sig': np.linspace(1, 5, num = 5), 'seed_grp': range(50), 'method': ['SVR', 'GB', 'RF'], 'size': [60, 80, 100]}
params_grid = list(ParameterGrid(params))
sig = params_grid[task_id]['sig'] * 3 / 10
seed_grp = params_grid[task_id]['seed_grp']
method= params_grid[task_id]['method']
k = params_grid[task_id]['size']

""" Generate data """
ntest = 100
set_id = 1
q = 0.1

theta = np.zeros(50).reshape((50,1))
theta[1:5,] = 0.1
all_res = pd.DataFrame()
seed_num = 20
batch_size = 10


for i in range(seed_num):
    seed = seed_grp * seed_num + i
    np.random.seed(seed)
    
    ## training data
    Xtrain, Ytrain, mu_train = gen_data(set_id, k, sig)
    Xcalib, Ycalib, mu_calib = gen_data(set_id, 200-k, sig)
    Xtest, Ytest, mu_test = gen_data(set_id, ntest, sig)

    """ ADACS """
    acs_obj = acs(q, 0, [method])
    acs_res = acs_obj.filter(Xtrain, Ytrain, Xcalib, Ycalib, Xtest, batch_size, prescreen=False)
    acs_res = acs_res.astype(bool)
    if np.sum(acs_res) == 0:
        acs_fdp = 0
        acs_power = 0
    else:
        acs_fdp = np.sum(Ytest[acs_res] <= 0) / np.sum(acs_res)
        acs_power = np.sum(Ytest[acs_res] > 0) / sum(Ytest > 0)
    
    all_res = pd.concat((all_res, 
                            pd.DataFrame({'fdp': [acs_fdp], 
                                        'power': [acs_power],
                                        'method': ['adacs'], 
                                        'seed': [seed]})))
    

all_res.to_csv(git_root + "/results/calibsize_setting"+str(set_id)+"k"+str(k)+"sig"+str(sig)+method+"seed"+str(seed_grp)+".csv")

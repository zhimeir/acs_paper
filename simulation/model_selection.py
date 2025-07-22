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
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression 
from sklearn.svm import SVR
from utils.cs_utils import gen_data, BH
from utils.acs import *

""" Configurations of the current run """
parser = argparse.ArgumentParser('')
parser.add_argument('--task_id', type=int, default=1)
args = parser.parse_args()
task_id = args.task_id - 1 
params = {'setting': [1, 2, 3, 4, 5], 'sig': np.linspace(1, 5, num = 5), 'seed_grp': range(50), 'size': [200, 500, 1000]}
params_grid = list(ParameterGrid(params))
set_id = params_grid[task_id]['setting']
if set_id == 5:
    sig = params_grid[task_id]['sig'] * 1 / 20
else:
    sig = params_grid[task_id]['sig'] * 3 / 10
seed_grp = params_grid[task_id]['seed_grp']
n = params_grid[task_id]['size']
mdl_list = ['SVR', 'GB', 'RF']

""" Parameters """
ntest = 100
q = 0.1
theta = np.zeros(50).reshape((50,1))
theta[1:5,] = 0.1
all_res = pd.DataFrame()
seed_num = 20
for i in range(seed_num):
    seed = seed_grp * seed_num + i
    np.random.seed(seed)

    ## Generating data
    Xtrain, Ytrain, mu_train = gen_data(set_id, n, sig)
    Xcalib, Ycalib, mu_calib = gen_data(set_id, n, sig)
    Xtest, Ytest, mu_test = gen_data(set_id, ntest, sig)

    """ ADACS """
    acs_obj = acs(q, 0, mdl_list)
    acs_res = acs_obj.filter(Xtrain, Ytrain, Xcalib, Ycalib, Xtest, batch_size = 20, prescreen=False)
    acs_res = acs_res.astype(bool)
    if sum(acs_res) == 0:
        acs_res_fdp = 0
        acs_res_power = 0
    else:
        acs_res_fdp = np.sum(Ytest[acs_res] <= 0) / sum(acs_res)
        acs_res_power = np.sum(Ytest[acs_res] > 0) / sum(Ytest > 0)
    all_res = pd.concat((all_res, 
                            pd.DataFrame({'fdp': [acs_res_fdp], 
                                        'power': [acs_res_power],
                                        'method': ['acs'], 'seed': [seed]})))
    """ CS """
    mse = np.inf
    # training the prediction model
    for method in mdl_list:
        
        if method == 'SVR':
            regressor = SVR(kernel="rbf", gamma=0.1)
        if method == 'GB': 
            regressor = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=0)
        if method == 'RF':
            regressor = RandomForestRegressor(max_depth=5, random_state=0)
        
        regressor.fit(Xtrain, 1*(Ytrain>0))
        mse_new = np.mean((1*(Ytrain>0) - regressor.predict(Xtrain))**2)
        
        # calibration 
        calib_scores_2clip = 1000 * (Ycalib > 0) - regressor.predict(Xcalib)
        test_scores = - regressor.predict(Xtest) 

        BH_2clip = BH(calib_scores_2clip, test_scores, q )
        if len(BH_2clip) == 0:
            BH_2clip_fdp = 0
            BH_2clip_power = 0
        else:
            BH_2clip_fdp = np.sum(Ytest[BH_2clip] <= 0) / len(BH_2clip)
            BH_2clip_power = np.sum(Ytest[BH_2clip] > 0) / sum(Ytest > 0)
        
        if mse_new < mse:
            winner = method
            mse = mse_new
            winner_fdp = BH_2clip_fdp
            winner_power = BH_2clip_power

        all_res = pd.concat((all_res, 
                                pd.DataFrame({'fdp': [BH_2clip_fdp], 
                                            'power': [BH_2clip_power],
                                            'method': [method], 'seed': [seed]})))
        

    all_res = pd.concat((all_res, 
                            pd.DataFrame({'fdp': [winner_fdp], 
                                        'power': [winner_power],
                                        'method': ['naive'], 'seed': [seed]})))

all_res.to_csv(git_root+"/results/model_selection_setting"+str(set_id)+"n"+str(n)+"sig"+str(sig)+"seed"+str(seed_grp)+".csv")

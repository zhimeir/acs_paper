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
params = {'sig': np.linspace(1, 10, num = 10), 'seed_grp': range(50), 'method': ['GB'], 'size': [500,1000], 'lamdiv': [0.3, 0.4, 0.5]}
params_grid = list(ParameterGrid(params))
sig = params_grid[task_id]['sig'] * 3 / 10
seed_grp = params_grid[task_id]['seed_grp']
method= params_grid[task_id]['method']
n = params_grid[task_id]['size']
lamdiv = params_grid[task_id]['lamdiv']

""" Generate data """
ntest = 200
set_id = 1
q = 0.1
es_sig = 5


theta = np.zeros(50).reshape((50,1))
theta[1:5,] = 0.1
all_res = pd.DataFrame()
seed_num = 20
batch_size = 10



for i in range(seed_num):
    seed = seed_grp * seed_num + i
    np.random.seed(seed)
    
    ## training data
    Xtrain, Ytrain, mu_train = gen_data(set_id, n, sig)
    Xcalib, Ycalib, mu_calib = gen_data(set_id, n, sig)
    Xtest, Ytest, mu_test = gen_data(set_id, ntest, sig)

    """ ADACS """
    adacs_obj = acs(q, 0, [method], div_index = 0, div_sig = es_sig)
    adacs_res = adacs_obj.filter(Xtrain, Ytrain, Xcalib, Ycalib, Xtest, batch_size = batch_size, prescreen = False)
    adacs_res = adacs_res.astype(bool)
    if np.sum(adacs_res) == 0:
        adacs_fdp = 0
        adacs_power = 0
        adacs_es = 2 
        adacs_num = 0
    else:
        adacs_fdp = np.sum(Ytest[adacs_res] <= 0) / np.sum(adacs_res)
        adacs_power = np.sum(Ytest[adacs_res] > 0) / sum(Ytest > 0)
        adacs_num = np.sum(adacs_res)
        len_acs_res = np.sum(Ytest[adacs_res] > 0)
        if len_acs_res > 1: ## only compute es when at least 2 positive samples are selected 
            adacs_es = (adacs_obj.compute_es(Xtest[adacs_res,], Ytest[adacs_res]>0, sig = es_sig).sum() - len_acs_res) / len_acs_res / (len_acs_res - 1)
        else:
            adacs_es = 2 
    

    all_res = pd.concat((all_res, 
                            pd.DataFrame({'fdp': [adacs_fdp], 
                                        'power': [adacs_power],
                                        'method': ['adacs'], 
                                        'es': [adacs_es],
                                        'num': [adacs_num],
                                        'seed': [seed]})))
    
    """ ADACS-div """
    div_obj = acs(q, 0, [method], div_index = lamdiv, div_sig = es_sig)
    div_res = div_obj.filter(Xtrain, Ytrain, Xcalib, Ycalib, Xtest, batch_size = batch_size)
    div_res = div_res.astype(bool)
    if np.sum(div_res) == 0:
        div_fdp = 0
        div_power = 0
        div_es = 2 
        div_num = 0
    else:
        div_fdp = np.sum(Ytest[div_res] <= 0) / np.sum(div_res)
        div_power = np.sum(Ytest[div_res] > 0) / sum(Ytest > 0)
        div_num = np.sum(div_res)
        len_div_res = np.sum(Ytest[div_res] > 0)
        if len_div_res > 1:
            div_es = (div_obj.compute_es(Xtest[div_res,], Ytest[div_res]>0, sig = es_sig).sum() - len_div_res) / len_div_res / (len_div_res - 1)
        else:
            div_es = 2

    

    all_res = pd.concat((all_res, 
                            pd.DataFrame({'fdp': [div_fdp], 
                                        'power': [div_power],
                                        'method': ['adacs.div'], 
                                        'es': [div_es],
                                        'num': [div_num],
                                        'seed': [seed]})))
    
   
    """ CS """
    # training the prediction model
    if method == 'SVR':
        regressor = SVR(kernel="rbf", gamma=0.1)
    if method == 'GB': 
        regressor = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=0)
    if method == 'RF':
        regressor = RandomForestRegressor(max_depth=5, random_state=0)
    regressor.fit(Xtrain, 1*(Ytrain>0))
        
    # calibration 
    calib_scores_2clip = 1000 * (Ycalib > 0) - regressor.predict(Xcalib)
    test_scores = - regressor.predict(Xtest) 

    BH_2clip = BH(calib_scores_2clip, test_scores, q )
    if len(BH_2clip) == 0:
        BH_2clip_fdp = 0
        BH_2clip_power = 0
        BH_2clip_es = 2
        BH_2clip_num = 0
    else:
        BH_2clip_fdp = np.sum(Ytest[BH_2clip] <= 0) / len(BH_2clip)
        BH_2clip_power = np.sum(Ytest[BH_2clip] > 0) / sum(Ytest > 0)
        BH_2clip_num = len(BH_2clip) 
        X_remain = Xtest[BH_2clip,]
        Y_remain = Ytest[BH_2clip]
        len_bh_res = np.sum(Ytest[BH_2clip] > 0)
        if len_bh_res > 1:
            BH_2clip_es = (adacs_obj.compute_es(X_remain,Y_remain>0, sig = es_sig).sum() - len_bh_res) / len_bh_res / (len_bh_res - 1)
        else: 
            BH_2clip_es = 2

    all_res = pd.concat((all_res, 
                            pd.DataFrame({'fdp': [BH_2clip_fdp], 
                                        'power': [BH_2clip_power],
                                        'method': ['CS'], 
                                        'es': [BH_2clip_es],
                                        'num': [BH_2clip_num], 
                                        'seed': [seed]})))


all_res.to_csv(git_root+"/results/diversity_setting"+str(set_id)+"n"+str(n)+"sig"+str(sig)+"lam"+str(lamdiv)+method+"seed"+str(seed_grp)+".csv")

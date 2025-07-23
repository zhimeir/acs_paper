import torch

import sys
# sys.path.append("..")
from importlib import reload
import persist_to_disk as ptd
import os
ptd.config.set_project_path(os.path.abspath("."))
import tqdm
import pandas as pd
import numpy as np
import re
import utils
import seaborn as sns
import math
import random

from utils.acs import *

from xgboost import XGBRFClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
from sklearn.linear_model import LogisticRegression

sns.set_theme()
sns.set_context("notebook")

import pipeline.clustering as pc
import pipeline.eval_uq as eval_uq

import argparse

import pickle
import pathlib

from _settings import GEN_PATHS

import matplotlib.pyplot as plt
# %matplotlib inline


import pipeline.uq_bb as uq_bb
reload(uq_bb)


params = {'model': ['llama-2-13b-chat-hf', 'opt-13b'], 'data': ['triviaqa', 'coqa'], 'N': [100, 500, 1000], 'split': [1,3], 'split_tune': [2]}
params_grid = list(ParameterGrid(params))


parser = argparse.ArgumentParser()
parser.add_argument('--repN', type=int, default=20)
parser.add_argument('--task_id', type=int, default=1)
args = parser.parse_args()


repN = args.repN
task_id = args.task_id - 1


model = params_grid[task_id]['model']
data = params_grid[task_id]['data']
split = params_grid[task_id]['split']
split_tune = params_grid[task_id]['split_tune']
N = params_grid[task_id]['N']

batch_size = 20
if data == 'triviaqa':
    batch_num = 400
else:
    batch_num = 385

print([model, data])


q_seq = np.round(np.linspace(0.05,0.95,19),3)
split_pr = split / 10
num_gens = 20
acc_name = 'generations|rougeL|acc'

path = GEN_PATHS[data][model] 

summ_kwargs = {
    'u+ea': {'overall': True, 'use_conf': False},
    'u+ia': {'overall': False, 'use_conf': False},
    'c+ia': {'overall': False, 'use_conf': True},
}['c+ia']

uq_list = [
        'generations|numsets', 
        'lexical_sim',
        'generations|spectral_eigv_clip|disagreement_w',
        'generations|eccentricity|disagreement_w',
        'generations|degree|disagreement_w',
        'generations|spectral_eigv_clip|agreement_w',
        'generations|eccentricity|agreement_w',
        'generations|degree|agreement_w',
        'generations|spectral_eigv_clip|jaccard',
        'generations|eccentricity|jaccard',
        'generations|degree|jaccard',
        'semanticEntropy|unnorm', 
        'self_prob',
]

mdl_list = ['rf', 'logistic', 'xgbrf']
rf_depth = 30
split_pr_tune = split_tune / 10

for method in mdl_list:


    if split_pr_tune < 1 - split_pr:


        # SAMPLE SIZE

        tune_size = math.floor(split_pr_tune * N)  # for choosing hyperparameters
        train_size = math.floor(split_pr * N)
        cal_size = N - tune_size - train_size
        test_size = 1000

        print([tune_size, train_size, cal_size])

        #################################################################
        #################################################################
        
        # reference kwargs
        o = uq_bb.UQ_summ(path, batch_num=batch_num, batch_size=batch_size, clean=True, split='test', cal_size=tune_size, train_size=train_size, seed=0)

        uq_kwargs_ref = summ_kwargs

        if len(o.key) > 2:
            assert o.key[2] == 'test'
            self2 = o.__class__(o.path, o.batch_num, o.batch_size, o.key[1], 'val', o.key[3], o.key[3], o.key[5])
            self2.tunable_hyperparams = {k:v for k, v in o.tunable_hyperparams.items() if k in uq_list}
            tuned_hyperparams = self2._tune_params(num_gens=num_gens,
                                                   metric=acc_name,
                                                   overall=False, use_conf=True, curve='auarc')
            uq_kwargs_ref.update(tuned_hyperparams)
        else:
            uq_kwargs_ref.update(o._get_default_params())

        print(f'uq_kwargs_ref: {uq_kwargs_ref}')


        par_path = '/'.join(path.split('/')[:-1])

        filename = os.path.join(par_path, 'uq_result/result')
        if not os.path.isdir(filename):
            os.makedirs(filename)


        # extract scores

        # overall_res, individual_res = o.get_uq_all(name='', uq_list=uq_list, acc_name=acc_name, num_gens=num_gens, uq_kwargs_ref=uq_kwargs_ref, uq_kwargs=uq_kwargs, test=False)

        # scores = (1-individual_res.to_numpy())
        # labels = o.get_acc(acc_name=acc_name, test=False)[1].to_numpy()

        #################################################################
        #################################################################


        uq_res = []
        for uq_ in uq_list:
            _, individual_res = o.get_uq(name=uq_, num_gens=num_gens, **uq_kwargs_ref.get(uq_,{}))
            print(individual_res.to_numpy().shape)
            uq_res.append(individual_res.to_numpy())
        
        print(uq_res.index)
        all_ids = o.ids

        uq_res = np.array(uq_res)
        uq_res = np.swapaxes(uq_res,0,1)
        print(f'shape of uq_res: {uq_res.shape}')

        label = o.get_acc(acc_name)[1]
        print(label.shape)


        if not os.path.isdir(f"./acs_aug_results/save_scores"):
            os.makedirs(f"./acs_aug_results/save_scores")
        if not os.path.exists(f'./acs_aug_results/save_scores/scores_{model}_{data}_{method}_{str(N)}_{str(split)}_{str(split_tune)}.npy'):
            np.save(f'./acs_aug_results/save_scores/scores_{model}_{data}_{method}_{str(N)}_{str(split)}_{str(split_tune)}.npy', uq_res)
            np.save(f'./acs_aug_results/save_scores/labels_{model}_{data}_{method}_{str(N)}_{str(split)}_{str(split_tune)}.npy', np.array(label))

        # pd.DataFrame(uq_res[:,:,0]).to_csv(f'./ada_results/scores.csv', index=True)  
        # pd.DataFrame(label).to_csv(f'./ada_results/labels.csv', index=False)

        for q in q_seq:

            all_res = pd.DataFrame()

            for seed in tqdm.tqdm(range(repN)):

                train_set = np.random.choice(uq_res.shape[0], train_size, replace=False)
                test_set = set(np.arange(uq_res.shape[0])) - set(train_set)
                train_ids = [all_ids[_] for _ in train_set]
                test_ids = [all_ids[_] for _ in test_set]

                cal_idx = np.random.choice(len(test_ids), cal_size, replace=False)
                test_idx = list(set(range(len(test_ids))) - set(cal_idx))

                train_label = label.loc[train_ids,:].to_numpy(dtype=int)
                test_label = label.loc[test_ids,:].to_numpy(dtype=int)
                uq_score = []

                ids_gen = 0
                Xtrain = uq_res[train_set,:,ids_gen]
                Ytrain = train_label[:,ids_gen]
                Xres = uq_res[list(test_set),:,ids_gen]
                Yres = test_label[:,ids_gen]

                Xcalib = Xres[cal_idx,:]
                Ycalib = Yres[cal_idx]
                Xtest = Xres[test_idx,:]
                Ytest = Yres[test_idx]

                test_idx = np.random.choice(len(Ytest), test_size, replace=False)
                Xtest = Xtest[test_idx,:]
                Ytest = Ytest[test_idx]

                print([Xtrain.shape, Xcalib.shape, Xtest.shape])
                print(np.mean(1.0*(Ytrain>0)))


                """ ADACS-aug """
                adacs_obj = acs(q, 0, [method], self_train = False)
                adacs_res = adacs_obj.filter_aug(Xtrain, Ytrain, Xcalib, Ycalib, Xtest, Ytest, batch_size=10)
                adacs_res = adacs_res.astype(bool)
                if np.sum(adacs_res) == 0:
                    adacs_res_fdp = 0
                    adacs_res_power = 0
                else:
                    adacs_res_fdp = np.sum(Ytest[adacs_res] <= 0) / np.sum(adacs_res)
                    adacs_res_power = np.sum(Ytest[adacs_res] > 0) / sum(Ytest > 0)

                all_res = pd.concat((all_res, 
                                        pd.DataFrame({'fdp': [adacs_res_fdp], 
                                                    'power': [adacs_res_power],
                                                    'method': ['acsaug'], 
                                                    'seed': [seed]})))

                """ ADACS """
                adacs_obj_0 = acs(q, 0, [method], self_train = False)
                # adacs_obj = adacs(q, 0, [method], self_train = False)
                adacs_res_0 = adacs_obj_0.filter(Xtrain, Ytrain, Xcalib, Ycalib, Xtest, batch_size = 10)
                adacs_res_0 = adacs_res_0.astype(bool)
                if sum(adacs_res_0) == 0:
                    adacs_res_fdp_0 = 0
                    adacs_res_power_0 = 0
                else:
                    adacs_res_fdp_0 = np.sum(Ytest[adacs_res_0] <= 0) / sum(adacs_res_0)
                    adacs_res_power_0 = np.sum(Ytest[adacs_res_0] > 0) / sum(Ytest > 0)
                all_res = pd.concat((all_res, 
                                        pd.DataFrame({'fdp': [adacs_res_fdp_0], 
                                                    'power': [adacs_res_power_0],
                                                    'method': ['adacs'], 'seed': [seed]})))


                """ CS """
                mse = np.inf
                # training the prediction model
                    
                if np.mean(1.0*(Ytrain>0)) == 0:
                    sc_calib = np.zeros(Xcalib.shape[0])
                    sc_test = np.zeros(Xtest.shape[0])
                elif np.mean(1.0*(Ytrain>0)) == 1:
                    sc_calib = np.ones(Xcalib.shape[0])
                    sc_test = np.ones(Xtest.shape[0])
                else:
                    if method == 'rf':
                        regressor = RandomForestClassifier(max_depth=rf_depth, random_state=2024)
                    
                    if method == 'logistic':
                        regressor = LogisticRegression(random_state=0)
                        
                        
                    if method == 'xgbrf':
                        regressor = XGBRFClassifier(n_estimators=100, subsample=0.9, colsample_bynode=0.2)
                    
                    regressor.fit(Xtrain, 1*(Ytrain>0))

                    sc_calib = regressor.predict_proba(Xcalib)[:,1]
                    sc_test = regressor.predict_proba(Xtest)[:,1]

                # calibration 
                calib_scores = 2.0 * (Ycalib > 0) - sc_calib
                test_scores = - sc_test

                BH_cs = BH(calib_scores, test_scores, q)
                if len(BH_cs) == 0:
                    BH_cs_fdp = 0
                    BH_cs_power = 0
                else:
                    BH_cs_fdp = np.sum(Ytest[BH_cs] <= 0) / len(BH_cs)
                    BH_cs_power = np.sum(Ytest[BH_cs] > 0) / sum(Ytest > 0)

                all_res = pd.concat((all_res, 
                                        pd.DataFrame({'fdp': [BH_cs_fdp], 
                                                        'power': [BH_cs_power],
                                                        'method': ['CS'], 'seed': [seed]})))

        
            print(all_res)

            out_filename = f"./acs_aug_results/model{model}_data{data}/model{model}_data{data}_N{str(N)}_q{str(int(q*100))}_split{int(split)}_splittune{split_tune}_method{method}"
            if not os.path.isdir(f"./acs_aug_results/model{model}_data{data}"):
                os.makedirs(f"./acs_aug_results/model{model}_data{data}")
            all_res.to_csv(out_filename+".csv")

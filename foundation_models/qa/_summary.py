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

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='opt-13b')
parser.add_argument("--batch_num", default=None, type=int, required=True,
                        help="Number of batches")
parser.add_argument("--idx", default=None, type=int, required=False,
                        help="Index of sequences of questions")
parser.add_argument("--batch_size", default=None, type=int, required=True,
                    help="Number of questions in each batch")
parser.add_argument('--data', type=str, default='coqa')
parser.add_argument("--cal_size", default=None, type=int, required=True,
                    help="Size of calibration set")
parser.add_argument("--train_size", default=None, type=int, required=True,
                    help="Size of training set (alignment)")
args = parser.parse_args()

model = args.model
data = args.data
batch_num = args.batch_num
batch_size = args.batch_size
cal_size = args.cal_size
train_size = args.train_size
num_gens = 20
acc_name = 'generations|rougeL|acc'
# acc_name='generations|gpt|acc'

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

uq_list_app = [
        'clf-rf',
        'clf-logistic',
        'clf-xgbrf',
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

o = uq_bb.UQ_summ(path, batch_num=batch_num, batch_size=batch_size, clean=True, split='test', cal_size=cal_size, train_size=train_size, seed=0)

# reference kwargs
uq_kwargs_ref = summ_kwargs

if len(o.key) > 2:
    assert o.key[2] == 'test'
    self2 = o.__class__(o.path, o.batch_num, o.batch_size, o.key[1], 'val', o.key[3], o.key[4], o.key[5])
    self2.tunable_hyperparams = {k:v for k, v in o.tunable_hyperparams.items() if k in uq_list}
    tuned_hyperparams = self2._tune_params(num_gens=num_gens,
                                           metric=acc_name,
                                           overall=False, use_conf=True, curve='auarc')
    uq_kwargs_ref.update(tuned_hyperparams)
else:
    uq_kwargs_ref.update(o._get_default_params())

print(f'uq_kwargs_ref: {uq_kwargs_ref}')

# kwargs
uq_kwargs = summ_kwargs

if len(o.key) > 2:
    assert o.key[2] == 'test'
    self2 = o.__class__(o.path, o.batch_num, o.batch_size, o.key[1], 'val', o.key[3], o.key[4], o.key[5])
    self2.tunable_hyperparams = {k:v for k, v in o.tunable_hyperparams.items() if k in uq_list_app and k in uq_list}
    tuned_hyperparams = self2._tune_params(num_gens=num_gens,
                                           metric=acc_name,
                                           overall=False, use_conf=True, curve='auarc')
    uq_kwargs.update(tuned_hyperparams)
else:
    uq_kwargs.update(o._get_default_params())

print(f'uq_kwargs: {uq_kwargs}')

par_path = '/'.join(path.split('/')[:-1])


# for uq in tqdm.tqdm(uq_list_app):
#     print(f'=====================================\n{uq}\n\n\n')

#     overall_res, individual_res = o.get_uq_all(name=uq, uq_list=uq_list, acc_name=acc_name, num_gens=20, uq_kwargs_ref=uq_kwargs_ref, uq_kwargs=uq_kwargs)

#     print(f'{uq}:\nshape of overall quantities {overall_res.shape}\nshape of individual quantities {individual_res.shape}')

#     print(path)
#     print(par_path)
#     pathlib.Path(f'{par_path}/uq_result').mkdir(parents=True, exist_ok=True)
#     with open(f'{par_path}/uq_result/ind_{uq}_{cal_size}_{train_size}_{batch_num}_{batch_size}.pkl', 'wb') as outfile:
#         pickle.dump(individual_res, outfile)
#     with open(f'{par_path}/uq_result/all_{uq}_{cal_size}_{train_size}_{batch_num}_{batch_size}.pkl', 'wb') as outfile:
#         pickle.dump(overall_res, outfile)

# summary

summ_obj = o.summ(uq_names=uq_list_app,
    uq_list=uq_list,
    acc_name=acc_name,
    num_gens=num_gens,
    uq_kwargs_ref=uq_kwargs_ref, 
    uq_kwargs=uq_kwargs,
)


# U + EA (using uncertainty to predict expected accuarcy)
summ_overall_auarc = summ_obj.summ_overall('auarc')
print(summ_overall_auarc)

# C + IA (using confidence to predict individual accuracy)
summ_ind_auarc = sum(summ_obj.summ_individual('auarc', use_conf=True)) / num_gens
print(summ_ind_auarc)

# C + IA (using confidence to predict individual accuracy)
summ_ind_auroc = sum(summ_obj.summ_individual('auroc', use_conf=True)) / num_gens
print(summ_ind_auroc)

summ_dict = {'overall_auarc': summ_overall_auarc, 'individual_auarc': summ_ind_auarc, 'individual_auroc': summ_ind_auroc}

pathlib.Path(f'{par_path}/uq_result').mkdir(parents=True, exist_ok=True)
with open(f'{par_path}/uq_result/summ_{cal_size}_{train_size}_{batch_num}_{batch_size}.pkl', 'wb') as outfile:
    pickle.dump(summ_dict, outfile)


# plot ROC curve 
plt.figure(figsize=(6, 3.5))
def name_map(v):
    if v == 'self_prob': return "P(true)"
    v = v.replace("|disagreement_w", "|(C)")
    v = v.replace("|agreement_w", "|(E)")
    v = v.replace("|jaccard", "|(J)")
    v = v.replace("spectral_eigv_clip|", "EigV")
    v = v.replace("eccentricity|", "Ecc")
    v = v.replace("degree|", "Deg")
    return {'numsets': 'NumSet', 'semanticEntropy|unnorm': 'SE',
            'blind': 'Basse Accuracy'}.get(v,v)
    return v
summ_obj.plot('roc', name_map=name_map, 
              methods=uq_list_app, 
              cutoff=1, iloc=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC")

plt.savefig(f'{par_path}/uq_result/roc_plot_{cal_size}_{train_size}.pdf', dpi=400, bbox_inches='tight')


# evaluate fdr control

def rt(lab_cal, sc_cal, sc_test, t):
    u = np.random.uniform(0,1,1)
    return ((u+np.sum((lab_cal==0) & (sc_cal>=t)))/(u+len(sc_cal))) * (len(sc_test) / max(1,np.sum((sc_test>=t))))


q = 0.2

for uq in uq_list:
    # uq = 'clf-rf'
    split_pr = 0.5
    num_thresholds = 50
    overall_res, individual_res = o.get_uq_all(name=uq, uq_list=uq_list, acc_name=acc_name, num_gens=20, uq_kwargs_ref=uq_kwargs_ref, uq_kwargs=uq_kwargs)
    scores = 1-individual_res.to_numpy()
    labels = o.get_acc(acc_name=acc_name, test=True)[1].to_numpy()

    t_seq = np.linspace(0, 1, num=num_thresholds)

    idx = 0
    scs = scores[:,idx]
    labs = labels[:,idx]

    cal_idx = (np.random.uniform(0,1,scores.shape[0]) <= split_pr)
    label_cal = labs[cal_idx]
    label_test = labs[~cal_idx]
    score_cal = scs[cal_idx]
    score_test = scs[~cal_idx]

    rt_seq = np.array([rt(label_cal, score_cal, score_test, t) for t in t_seq])
    idx_set = np.where(rt_seq <= q)[0]

    # tau = t_seq[idx_set[0]]

    individual_res.to_csv(f'{par_path}/uq_result/scores_{uq}.csv', index=True)  
    pd.DataFrame(labels).to_csv(f'{par_path}/uq_result/labels_{uq}.csv', index=False)

    # fdp = np.sum((label_test==0)&(score_test>=tau))/max(1,np.sum(score_test>=tau))
    # power = np.sum((label_test==1)&(score_test>=tau))/max(1,np.sum(label_test==1))
    # # print(r'threshold: %s'%(tau,))
    # print('false discovery proportion: %s\npower: %s'%(fdp,power))
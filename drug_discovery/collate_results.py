import numpy as np
import pandas as pd
import os
import sys
from tqdm import tqdm

print("here")
variants = []
for job in range(250):
   for quantile_indexer in range(6):
      for alpha_indexer in range(4):
        variants.append((job,quantile_indexer,alpha_indexer))

results_dict = dict()
quantiles = [0.2, 0.25, 0.3, 0.35]

dne = 0
for quantile_indexer in range(6):
   for alpha_indexer in range(4):
      for method in range(2):
         for metric in range(3):
            results_dict[quantile_indexer,alpha_indexer, method, metric] = []




for seed in tqdm(range(6000)):
   _, quantile_indexer, alpha_indexer = variants[seed]
    #print(f'metrics_c{couple}_s{setting}_j{job}.csv')
    #print(pd.read_csv(f'sharpe_results/metrics_c{couple}_s{setting}_j{job}.csv', header=False).to_numpy().astype(float))
   #  try:
   acs_results = pd.read_csv(f'dti_similarity_results/metrics_v{seed}.csv', header=None).to_numpy().astype(float)[0]
   for metric in range(3):
      results_dict[quantile_indexer, alpha_indexer, 0, metric].append(acs_results[metric])
   vanilla_results = pd.read_csv(f'dti_similarity_results/vanilla_metrics_v{seed}.csv', header=None).to_numpy().astype(float)[0]
   for metric in range(3):
      results_dict[quantile_indexer, alpha_indexer, 1, metric].append(vanilla_results[metric])
      
   #  except:
   #     print(f'Result doesnt exist for variant = {seed}')
   #     dne += 1

# for quantile_indexer in range(4):
#     print("----------------------------")
#     print(f'ACS FDR, Power, ESS for quantile = {quantiles[quantile_indexer]}: [{np.mean(results_dict[quantile_indexer,0,0])} (+/- {np.std(results_dict[quantile_indexer,0,0])/np.sqrt(len((results_dict[quantile_indexer,0,0])))}), {np.mean(results_dict[quantile_indexer,0,0])} (+/- {np.std(results_dict[quantile_indexer,0,1])/np.sqrt(len((results_dict[quantile_indexer,0,1])))}), {np.mean(results_dict[quantile_indexer,0,2])} (+/- {np.std(results_dict[quantile_indexer,0,2])/np.sqrt(len((results_dict[quantile_indexer,0,2])))})]')
#     print(f'CS FDR, Power, ESS for quantile = {quantiles[quantile_indexer]}: [{np.mean(results_dict[quantile_indexer,0,0])} (+/- {np.std(results_dict[quantile_indexer,0,0])/np.sqrt(len((results_dict[quantile_indexer,1,0])))}), {np.mean(results_dict[quantile_indexer,1,0])} (+/- {np.std(results_dict[quantile_indexer,1,1])/np.sqrt(len((results_dict[quantile_indexer,1,1])))}), {np.mean(results_dict[quantile_indexer,1,2])} (+/- {np.std(results_dict[quantile_indexer,1,2])/np.sqrt(len((results_dict[quantile_indexer,1,2])))})]')
#     print("----------------------------")


alphas = ['alpha=0.25', 'alpha=0.3', 'alpha=0.35', 'alpha=0.4']

alphas_for_R = []
quantiles_for_R = []
method_for_R = []
ess_for_R = []
fdr_for_R = []
power_for_R = []
ess_serr_for_R = []
fdr_serr_for_R = []
power_serr_for_R = []

for alpha_indexer in range(len(alphas)):
   for quantile_indexer in range(len(quantiles)):
      alphas_for_R.append(alphas[alpha_indexer])
      method_for_R.append("ACS")
      quantiles_for_R.append(f'Quantile = {quantiles[quantile_indexer]}')
      fdr_for_R.append(np.mean(results_dict[quantile_indexer,alpha_indexer,0,0]))
      fdr_serr_for_R.append(np.std(results_dict[quantile_indexer,alpha_indexer,0,0])/len(np.sqrt(results_dict[quantile_indexer,alpha_indexer,0,0])))
      power_for_R.append(np.mean(results_dict[quantile_indexer,alpha_indexer,0,1]))
      power_serr_for_R.append(np.std(results_dict[quantile_indexer,alpha_indexer,0,1])/len(np.sqrt(results_dict[quantile_indexer,alpha_indexer,0,1])))
      ess_for_R.append(np.mean(results_dict[quantile_indexer,alpha_indexer,0,2]))
      ess_serr_for_R.append(np.std(results_dict[quantile_indexer,alpha_indexer,0,2])/len(np.sqrt(results_dict[quantile_indexer,alpha_indexer,0,2])))

      alphas_for_R.append(alphas[alpha_indexer])
      method_for_R.append("CS")
      quantiles_for_R.append(f'Quantile = {quantiles[quantile_indexer]}')
      fdr_for_R.append(np.mean(results_dict[quantile_indexer,alpha_indexer,1,0]))
      fdr_serr_for_R.append(np.std(results_dict[quantile_indexer,alpha_indexer,1,0])/len(np.sqrt(results_dict[quantile_indexer,alpha_indexer,1,0])))
      power_for_R.append(np.mean(results_dict[quantile_indexer,alpha_indexer,1,1]))
      power_serr_for_R.append(np.std(results_dict[quantile_indexer,alpha_indexer,1,1])/len(np.sqrt(results_dict[quantile_indexer,alpha_indexer,1,1])))
      ess_for_R.append(np.mean(results_dict[quantile_indexer,alpha_indexer,1,2]))
      ess_serr_for_R.append(np.std(results_dict[quantile_indexer,alpha_indexer,1,2])/len(np.sqrt(results_dict[quantile_indexer,alpha_indexer,1,2])))


fdr_results = pd.DataFrame({
    'Method': method_for_R,
    'FDR': fdr_for_R,
    'quantile': quantiles_for_R,
    'label': ['FDR']*len(quantiles_for_R),
    'serr': fdr_serr_for_R,
    'alpha': alphas_for_R
})

fdr_results.to_csv('./csvs_to_plot/fdr_results.csv')

power_results = pd.DataFrame({
    'Method': method_for_R,
    'Power': power_for_R,
    'quantile': quantiles_for_R,
    'label': ['Power']*len(quantiles_for_R),
    'serr': power_serr_for_R,
    'alpha': alphas_for_R
    })

power_results.to_csv('./csvs_to_plot/power_results.csv')

ess_results = pd.DataFrame({
    'Method': method_for_R,
    'ESS': ess_for_R,
    'quantile': quantiles_for_R,
    'label': ['ESS']*len(quantiles_for_R),
    'serr': ess_serr_for_R,
    'alpha': alphas_for_R
    })

ess_results.to_csv('./csvs_to_plot/ess_results.csv')
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
import pandas as pd
import numpy as np
from tqdm import tqdm

from DeepPurpose import utils # , dataset, CompoundPred
from DeepPurpose import DTI as models
from DeepPurpose.utils import *
from DeepPurpose.dataset import *
from rdkit import Chem
from rdkit import DataStructs
#from rdkit.ML.Cluster import Butina
#from rdkit.Chem import Draw
#from rdkit.Chem import rdFingerprintGenerator
#from rdkit.Chem.Draw import SimilarityMaps
import warnings
import numpy as np
import sys 
import pandas as pd 
import cvxpy as cp
warnings.filterwarnings("ignore")

def softmax(x):
    return np.exp(x)/sum(np.exp(x))

variants = []
for job in range(250):
   for quantile_indexer in range(6):
      for alpha_indexer in range(4):
        variants.append((job,quantile_indexer,alpha_indexer))

seed = int(sys.argv[1])
acs_fdrs = []
acs_powers = []
bh_fdrs = []
bh_powers = []

def BH(calibS, testS, alpha = 0.1):
    combinedS = np.concatenate((calibS, testS))
    argsortedS = np.argsort(combinedS)

    n = len(calibS)
    m = len(testS)

    numCalibRejections = n
    numTestRejections = m
    FDP_hat = ((1.+numCalibRejections)/max(1.,numTestRejections))*(m/(n+1.))

    indexer = n+m-1
    while FDP_hat > alpha and indexer >= 0:
        #print(FDP_hat)
        numCalibRejections -= int(argsortedS[indexer] < n)
        numTestRejections -= int(argsortedS[indexer] >= n)
        FDP_hat = ((1.+numCalibRejections)/max(1.,numTestRejections))*(m/(n+1.))
        indexer -= 1

    if indexer < 0:
        return np.zeros(m)
    
    return (testS <= combinedS[argsortedS[indexer]]).astype(bool)

def filter(ttrain, dcalib, dtest, similarityMatrix, alpha, div_index = 0.4, c = 0):
    """Configurations"""
    print("INSIDE FILTER")
    n = len(dcalib)
    m = len(dtest)

    batch_size = 10

    config = utils.generate_config(drug_encoding = drug_encoding, 
                        target_encoding = target_encoding, 
                        cls_hidden_dims = [1024,1024,512], 
                        train_epoch = 10, 
                        LR = 0.001, 
                        batch_size = 128,
                        hidden_dim_drug = 128,
                        mpnn_hidden_size = 128,
                        mpnn_depth = 3, 
                        cnn_target_filters = [32,64,96],
                        cnn_target_kernels = [4,8,12]
                        )


    model = models.model_initialize(**config)

    #model.train(ttrain)

    """Initialization"""
    sc_rec = np.ones(n + m)  # 1 = not screened, 0 = screened
    cal_id = np.ones(n + m)
    cal_id[n:] = 0
    test_id = np.zeros(n + m)
    test_id[n:] = 1
    count = 0
    Y_calib = dcalib['Label'].values
    Y_all = np.concatenate([Y_calib, c[n:]])

    #print(len(Y_calib), len(Y_all), len(sc_rec), n, m)

    # Initial hFDP
    hfdp = (1 + np.sum(sc_rec * cal_id * (Y_all <= c))) / np.sum(sc_rec * test_id) * m / (1 + n)
    record = []
    """Sequential screening"""
    while hfdp > alpha and np.sum(sc_rec * test_id) > 0:
        print(hfdp)
        # X_sc = X_calib[(1-sc_rec).astype(bool)[:n], :]
        # Y_sc = Y_calib[(1-sc_rec).astype(bool)[:n]]
        # X_usc = X_calib[sc_rec.astype(bool)[:n], :]
        # X_remain = np.concatenate([X_usc, X_test])

        calib_sc = dcalib.iloc[(1-sc_rec).astype(bool)[:n]]

        if count % batch_size == 0:
            """Update the model"""
            print("Go for training...")
            model.train(pd.concat((ttrain,calib_sc)).reset_index(drop=True))
            count += 1

            Yhat_test = softmax(model.predict(pd.concat([dcalib, dtest]).reset_index(drop=True)))
            #record.append(Yhat_test[n:] * 1.0)

            remaining_similarity_matrix = similarityMatrix[sc_rec.astype(bool)][:,sc_rec.astype(bool)]
            if div_index > 0:
                #X_remain = np.concatenate([X_calib, X_test])[sc_rec == 1, :]
                Yhat_remain = Yhat_test[sc_rec == 1] * 1.0
                es = remaining_similarity_matrix*Yhat_remain*Yhat_remain[:,np.newaxis]
                
                print("cvxpy start...")
                #constraint_matrix = np.vstack((np.ones(es.shape[0]), Yhat_remain))
                #constraint_equality = np.array([1./(1-alpha), 1.])
                xi = cp.Variable(es.shape[0])
                constraints = [xi >= 0,
                               xi.T @ Yhat_remain == 1]#./es.shape[0],
                              #xi.T @ np.ones(es.shape[0]) <= 1./(1-alpha)]
                obj = cp.Minimize(cp.quad_form(xi, cp.psd_wrap(es)))
                prob = cp.Problem(obj, constraints)
                prob.solve(verbose=False, solver=cp.MOSEK)
                print("Status:", prob.status)
                div_prob = xi.value
                print("cvxpy end...")
                # print(div_prob)
                # print(np.linalg.cholesky(es))
                # print(alpha)
                # print(Yhat_remain)

                # print(f'poss soln: {np.all(np.linalg.pinv(constraint_matrix)@constraint_equality >= 0)}')
                # print(constraint_matrix @ np.linalg.pinv(constraint_matrix) @constraint_equality == constraint_equality)
                
                # es_inv = np.linalg.inv(es)
                # base1 = np.dot(es_inv, Yhat_remain)
                # base2 = np.dot(es_inv, np.ones(len(Yhat_remain)))
                # AA = np.dot(Yhat_remain, base2)
                # BB = np.dot(Yhat_remain, base1)
                # CC = np.dot(np.ones(len(Yhat_remain)), base2)
                # div_prob = (-AA + BB / (1 - alpha)) * base2 - (AA / (1 - alpha) - CC) * base1
                Yhat_test[sc_rec == 1] = (1 - div_index) * Yhat_remain + div_index * div_prob / np.max(div_prob)

        usc_id = np.where(sc_rec > 0)[0]
        Yhat_usc = Yhat_test[usc_id]
        next_id0 = np.random.choice(np.where(Yhat_usc == np.min(Yhat_usc))[0])
        next_id = usc_id[next_id0]
        sc_rec[next_id] = 0

        if cal_id[next_id] == 1:
            count += 1

        if np.sum(sc_rec * test_id) == 0:
            break

        hfdp = (1 + np.sum(sc_rec * cal_id * (Y_all <= c))) / np.sum(sc_rec * test_id) * m / (1 + n)

    return (sc_rec * test_id)[n:]

np.random.seed(seed)
_, quantile_indexer, alpha_indexer = variants[seed]

quantiles = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45]
alphas = [0.25, 0.3, 0.35, 0.4]

alpha = alphas[alpha_indexer]

drug_encoding = 'CNN'
target_encoding = 'Transformer'


print("loading...")
full_similarity_matrix = np.load('similarityMatrix.npy')
df = pd.read_csv('cleaned_drug_data.csv')
print("finished loading...")

affinity_list = df['affinity'].tolist()
ligand_smiles_list = df['ligand'].tolist()
target_aac_list = df['target'].tolist()

n_total = len(affinity_list)
reind = np.random.permutation(n_total)

frac = 1./4

X_drugs_train = [ligand_smiles_list[reind[0:int(n_total*frac+1)][i]] for i in range(int(n_total*frac+1))]
X_targets_train = [target_aac_list[reind[0:int(n_total*frac+1)][i]] for i in range(int(n_total*frac+1))]
y_train = [affinity_list[reind[0:int(n_total*frac+1)][i]] for i in range(int(n_total*frac+1))]


X_drugs_other = [ligand_smiles_list[reind[int(1+n_total*frac):n_total][i]] for i in range(n_total-int(n_total*frac+1))]
X_targets_other = [target_aac_list[reind[int(1+n_total*frac):n_total][i]] for i in range(n_total-int(n_total*frac+1))]
y_other = [affinity_list[reind[int(1+n_total*frac):n_total][i]] for i in range(n_total-int(n_total*frac+1))]


print("processing...")
ttrain, tval, ttest = utils.data_process(X_drugs_train, X_targets_train, y_train, 
                                drug_encoding, target_encoding, 
                                split_method='random', frac=[1.,0.,0.],
                                random_seed = seed)

ddata, _, __ = utils.data_process(X_drugs_other, X_targets_other, y_other, 
                                drug_encoding, target_encoding, 
                                    split_method='random', frac=[1., 0., 0.],
                                    random_seed = seed)
print("finished processing...")


calib_test_perm = np.random.permutation(len(ddata))
calib_test_ratio = 1./3#2./3
dcalib = ddata.iloc[calib_test_perm[0:int(len(ddata)*calib_test_ratio+1)]].reset_index(drop=True)
dtest = ddata.iloc[calib_test_perm[int(1+len(ddata)*calib_test_ratio):len(ddata)]].reset_index(drop=True)
ytest = dtest['Label'].values
#dtest = dtest.drop('Label', axis=1)


print("***")
print(len(dcalib), len(dtest))
print("***")

final_perm = np.array(reind[int(1+n_total*frac):n_total])[calib_test_perm]
similarityMatrix = full_similarity_matrix[final_perm][:,final_perm]


# get quantile cutoffs
testq = np.zeros(dtest.shape[0])


for i in range(dtest.shape[0]):
    tenc = dtest['Target Sequence'].iloc[i]
    tsub = ttrain['Target Sequence'] == tenc 
    if sum(tsub) == 0:
        allb = ttrain 
    else:
        allb = ttrain[tsub]

    quantile = np.quantile(allb['Label'], quantiles[quantile_indexer])
    testq[i] = quantile


calibq = np.zeros(dcalib.shape[0])


for i in range(dcalib.shape[0]):
    tenc = dcalib['Target Sequence'].iloc[i]
    tsub = ttrain['Target Sequence'] == tenc
    # print(tsub)
    if sum(tsub) == 0:
        allb = ttrain 
    else:
        allb = ttrain[tsub]

    # allb = ttrain[]
    # print(allb['Label'])
    # print(allb)
    quantile = np.quantile(allb['Label'], quantiles[quantile_indexer])
    calibq[i] = quantile


n = len(dcalib)

print(len(ttrain),len(dcalib),len(dtest))

print("Running acs...")
acs_rej = filter(ttrain, dcalib, dtest, similarityMatrix, alpha, div_index = 0.8, c = np.concatenate((calibq, testq)))

acs_fdp = np.sum((ytest <= testq)*acs_rej)/max(1.,np.sum(acs_rej))
acs_power = np.sum((ytest > testq)*acs_rej)/max(1.,np.sum(ytest > testq))
acs_similarity = ((ytest > testq)*acs_rej).T @ similarityMatrix[n:][:,n:] @ ((ytest > testq)*acs_rej)\
/max(1., ((ytest > testq)*acs_rej).T @ np.ones((len(ytest), len(ytest))) @ ((ytest > testq)*acs_rej))

acs_fdrs.append(acs_fdp)
acs_powers.append(acs_power)



acs_results = [acs_fdp, acs_power, acs_similarity]



config = utils.generate_config(drug_encoding = drug_encoding, 
                        target_encoding = target_encoding, 
                        cls_hidden_dims = [1024,1024,512], 
                        train_epoch = 10, 
                        LR = 0.001, 
                        batch_size = 128,
                        hidden_dim_drug = 128,
                        mpnn_hidden_size = 128,
                        mpnn_depth = 3, 
                        cnn_target_filters = [32,64,96],
                        cnn_target_kernels = [4,8,12]
                        )


model = models.model_initialize(**config)
model.train(ttrain)


calibS = np.where(dcalib['Label'].values - calibq > 0, np.inf, - np.array(model.predict(dcalib)))
testS = -np.array(model.predict(dtest))

bh_rej = BH(calibS, testS, alpha)
bh_fdp = np.sum((ytest <= testq)*bh_rej)/max(1.,np.sum(bh_rej))
bh_power = np.sum((ytest > testq)*bh_rej)/max(1.,np.sum(ytest > testq))
bh_similarity = ((ytest > testq)*bh_rej).T @ similarityMatrix[n:][:,n:] @ ((ytest > testq)*bh_rej)\
/max(1., ((ytest > testq)*bh_rej).T @ np.ones((len(ytest), len(ytest))) @ ((ytest > testq)*bh_rej))


bh_results = [bh_fdp, bh_power, bh_similarity]


print(bh_fdp, bh_power,bh_rej.sum())
#print(f'Running means: {np.mean(acs_fdrs), np.mean(acs_powers)}')

with open(f"dti_similarity_results/metrics_v{seed}.csv", "at") as file:
    file.write(",".join(map(str, acs_results)) + "\n")

# with open(f"sharpe_results/vanilla_rejections_c{couple}_s{setting}_j{job}.csv", "at") as file:
#     file.write(",".join(map(str, vanilla_rejections)) + "\n")

with open(f"dti_similarity_results/vanilla_metrics_v{seed}.csv", "at") as file:
    file.write(",".join(map(str, bh_results)) + "\n")
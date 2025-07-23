import numpy as np
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR, SVC
from xgboost import XGBRFClassifier
from sklearn.model_selection import ParameterGrid
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from scipy.stats import randint
from sklearn.linear_model import LogisticRegression
import pandas as pd
from sklearn.model_selection import KFold



"""

changes based on zhimeir's version

Yhat_test = predicted prob

remove diversity part for now

"""





def my_argmin(arr):
    indices = np.where(arr == arr.min())[0]
    idx = np.random.choice(indices)
    return idx

class acs:
    def __init__(self, alpha, c, mdl_list = ['rf', 'logistic', 'xgbrf'], self_train = False, div = False, lamdiv = 1, sig = 1):
        self.alpha = alpha 
        self.c = c
        self.mdl_list = mdl_list
        self.regressor = mdl_list[0] # default regressor
        self.self_train = self_train
        self.div = div
        self.lamdiv = lamdiv
        self.sig = sig

    def model_update(self, X_train, Y_train, X_sc, Y_sc, X_usc):
        
        X_train_aug = np.concatenate((X_train, X_sc))
        Y_train_aug = np.concatenate((Y_train, Y_sc))

        if self.self_train:

            X_train_mixed = np.concatenate((X_train_aug, X_usc)) 
            nolabel = [-1 for _ in range(len(X_usc))]
            Y_train_mixed = np.concatenate((1.*(Y_train_aug>0), nolabel))

            if self.regressor == 'rf':
                rf_depth = 30
                base_model = RandomForestClassifier(max_depth=rf_depth, random_state=2024)
                self.model = SelfTrainingClassifier(base_estimator = base_model)

            if self.regressor == 'logistic':
                base_model = LogisticRegression(random_state=0)
                self.model = SelfTrainingClassifier(base_estimator = base_model)

            if self.regressor == 'xgbrf':
                base_model = XGBRFClassifier(n_estimators=100, subsample=0.9, colsample_bynode=0.2)
                self.model = SelfTrainingClassifier(base_estimator = base_model)
            
            self.model.fit(X_train_mixed, Y_train_mixed)

        else: 

            if self.regressor == 'rf':
                rf_depth = 30
                self.model = RandomForestClassifier(max_depth=rf_depth, random_state=2024)

            if self.regressor == 'logistic':
                self.model = LogisticRegression(random_state=0)

            if self.regressor == 'xgbrf':
                self.model = XGBRFClassifier(n_estimators=100, subsample=0.9, colsample_bynode=0.2)

            self.model.fit(X_train_aug, 1.*(Y_train_aug>0))


    def model_selection(self, X_train, Y_train, X_sc, Y_sc, X_usc, X_test): 

        X_train_aug = np.concatenate((X_train, X_sc))
        Y_train_aug = np.concatenate((Y_train, Y_sc))
        res_list = [] # record the metrics of different models
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        for method in self.mdl_list:
            
            method_res = 0
            for train_index, test_index in kfold.split(X_train_aug):
                X_train_k, X_test_k = X_train_aug[train_index,:], X_train_aug[test_index,:]
                Y_train_k, Y_test_k = Y_train_aug[train_index], Y_train_aug[test_index]
                
                if method == 'rf':
                    rf_depth = 30
                    model = RandomForestClassifier(max_depth=rf_depth, random_state=2024)
            
                if method == 'logistic':
                    model = LogisticRegression(random_state=0)
                
                
                if method == 'xgbrf':
                    model = XGBRFClassifier(n_estimators=100, subsample=0.9, colsample_bynode=0.2)
            
                # model.fit(X_train_k, 1.*(Y_train_k>0))
                # Yhat = model.predict(np.concatenate([X_test_k, X_usc, X_test]))
                
                # # prob_aug = model.predict_proba(X_train_aug)[:,1]
                # if np.mean(1.0*(Y_train_aug>0)) == 0:
                #     prob_aug = np.zeros(X_train_aug.shape[0])
                # elif np.mean(1.0*(Y_train_aug>0)) == 1:
                #     prob_aug = np.ones(X_train_aug.shape[0])
                # else:
                #     prob_aug = model.predict_proba(X_train_aug)[:,1]

                # model.fit(X_train_k, 1.*(Y_train_k>0))
                # Yhat = model.predict(np.concatenate([X_test_k, X_usc, X_test]))
                # prob_aug = model.predict_proba(X_train_aug)[:,1]

                X_ = np.concatenate([X_test_k, X_usc, X_test])
                if np.mean(1.0*(Y_train_k>0)) == 0:
                    Yhat = np.zeros(X_.shape[0])
                    prob_aug = np.zeros(X_train_aug.shape[0])
                elif np.mean(1.0*(Y_train_k>0)) == 1:
                    Yhat = np.ones(X_.shape[0])
                    prob_aug = np.ones(X_train_aug.shape[0])
                else:
                    model.fit(X_train_k, 1.*(Y_train_k>0))
                    Yhat = model.predict_proba(np.concatenate([X_test_k, X_usc, X_test]))[:,1]
                    prob_aug = model.predict_proba(X_train_aug)[:,1]


                if len(X_test_k) > 0:
                    method_res += np.sum(self.seqstep(Y_test_k, Yhat))
                else: 
                    # method_res += - np.mean((1*(Y_train_aug>0) - model.predict(X_train_aug)[:,1])**2)
                    method_res = - np.mean((1.*(Y_train_aug>0) - prob_aug)**2)
            
            res_list.append(np.mean(method_res))

        winner = np.argmax(res_list)
        self.regressor = self.mdl_list[winner]


    def filter(self, X_train, Y_train, X_calib, Y_calib, X_test, batch_size = 1, membership_mat = None, div_index = 0):
        
        """ configurations """
        n = len(X_calib)
        m = len(X_test)
        mdl_select = False
        if len(self.mdl_list) > 1: # if there are multiple models to choose from
            mdl_select = True 
        
        """ initialization """
        hfdp = 1
        sc_rec = np.ones(n+m) # record the screening results: 1 for not screened, 0 for screened
        cal_id = np.ones(n+m)
        cal_id[n:] = 0
        test_id = np.zeros(n+m)
        test_id[n:] = 1
        count = 0 # keep track of the batch size
        sel_id = np.ones(n) ## record the newly screened calibration points in the current batch 

        #if self.div:
        #self.es_const = self.compute_es(X_train,Y_train, sig=self.sig).mean()

        self.record = []     
        
        """ sequantial screening """
        while hfdp > self.alpha and np.sum(sc_rec * test_id) > 0:

            sc_cal = sc_rec * cal_id
            sc_cal = sc_cal[:n]
            X_sc = X_calib[sc_cal <= 0,:]
            Y_sc = Y_calib[sc_cal <= 0]
            X_osc = X_calib[(sc_cal <= 0) * (sel_id == 1),:] 
            Y_osc = Y_calib[(sc_cal <= 0) * (sel_id == 1)]
            X_nsc = X_calib[(sc_cal <= 0) * (sel_id == 0),:] ## newly screened calibration data points
            Y_nsc = Y_calib[(sc_cal <= 0) * (sel_id == 0)] ## newly screened calibration data points
            X_usc = X_calib[sc_cal > 0,:] ## unscreened calibration data points

            if(count % batch_size == 0): ## update the model every batch_size iterations 
                print(np.mean(1.0*(Y_train>0)))

                if np.mean(1.0*(Y_train>0)) == 0:
                    Yhat_test = np.zeros(np.concatenate([X_calib, X_test]).shape[0])
                elif np.mean(1.0*(Y_train>0)) == 1:
                    Yhat_test = np.ones(np.concatenate([X_calib, X_test]).shape[0])
                else:
                    if mdl_select: ## select the best model
                        self.model_selection(X_train, Y_train, X_sc, Y_sc, X_usc, X_test)
                    self.model_update(X_train, Y_train, X_sc, Y_sc, X_usc)

                    Yhat_test = self.model.predict_proba(np.concatenate([X_calib, X_test]))[:,1]

                    # if self.self_train: ## self-training
                    #     Yhat_test = self.model.predict_proba(np.concatenate([X_calib, X_test]))[:,1]
                    # else: ## standard training
                    #     Yhat_test = self.model.predict(np.concatenate([X_calib, X_test])) 
                
                self.record.append(Yhat_test[n:]*1.)

                # if self.div: ## diversity regularization 
                #     X_remain = np.concatenate([X_calib, X_test])[sc_rec == 1,:]
                #     Yhat_remain = Yhat_test[sc_rec == 1] * 1.
                #     self.es = self.compute_es(X_remain, Yhat_remain, sig = self.sig)
                #     es_inv = np.linalg.inv(self.es)
                #     base1 = np.dot(es_inv, Yhat_remain)
                #     base2 = np.dot(es_inv, np.ones(len(Yhat_remain)))
                #     AA = np.dot(Yhat_remain,base2)
                #     BB = np.dot(Yhat_remain,base1)
                #     CC = np.dot(np.ones(len(Yhat_remain)),base2)
                #     div_prob = (-AA + BB / (1-self.alpha)) * base2 - (AA / 1-self.alpha - CC) * base1
                #     Yhat_test[sc_rec == 1] = (1 -  self.lamdiv) * Yhat_remain + self.lamdiv * div_prob / np.max(div_prob)

                max_range = np.max(Yhat_test) + 100
                Yhat_test[sc_rec==0] = max_range
                
                
                # if div_index > 0:
                #     usc_mem = membership_mat[sc_rec > 0,:] ## membership matrix of the screened calibration data points
                #     group_size = np.sum(usc_mem, axis = 0)
                #     endangered_group = (group_size < div_index)
                #     endangered_id = np.sum(usc_mem[:,endangered_group], axis = 1)
                #     Yhat_test[sc_rec>0][endangered_id>0] = max_range * 1.
                    
                sel_id = np.ones(n)
            
            #next_id = np.argmin(Yhat_test)
            # next_id = np.random.choice(np.where(Yhat_test == np.min(Yhat_test))[0]) ## randomly choose one of the minimum values if there are ties
            next_id = my_argmin(Yhat_test)
            sc_rec[next_id] = 0
            Yhat_test[next_id] = max_range

            # ## enforce a (semi-)hard constraint on the minimum group size 
            # if div_index > 0:
            #     usc_mem = membership_mat[sc_rec > 0,:] ## membership matrix of the screened calibration data points
            #     group_size = np.sum(usc_mem, axis = 0)
            #     endangered_group = (group_size < div_index)
            #     endangered_id = np.sum(usc_mem[:,endangered_group], axis = 1)
            #     Yhat_test[sc_rec>0][endangered_id>0] = max_range * 1.
           
            if(cal_id[next_id] == 1):
                count +=1
                sel_id[next_id] = 0
            Y_all = np.concatenate([Y_calib, np.zeros(m)])
            if np.sum(sc_rec * test_id) == 0:
                break
            hfdp = (1 + np.sum(sc_rec * cal_id * (Y_all <= self.c))) / np.sum(sc_rec * test_id) * m / (1+n)

        return (sc_rec * test_id)[n:] 
    
    """ when screened test data points can be labeled """    
    def filter_aug(self, X_train, Y_train, X_calib, Y_calib, X_test, Y_test, batch_size = 1, membership_mat = None, div_index = 0):
        
        """ configurations """
        n = len(X_calib)
        m = len(X_test)
        X_aug = np.concatenate([X_calib, X_test])
        Y_aug = np.concatenate([Y_calib, Y_test])
        mdl_select = False
        if len(self.mdl_list) > 1: # if there are multiple models to choose from
            mdl_select = True 
        
        """ initialization """
        hfdp = 1
        sc_rec = np.ones(n+m) # record the screening results: 1 for not screened, 0 for screened
        cal_id = np.ones(n+m)
        cal_id[n:] = 0
        test_id = np.zeros(n+m)
        test_id[n:] = 1
        count = 0 # keep track of the batch size
        sel_id = np.ones(n+m) ## record the newly screened points in the current batch 

        """ sequantial screening """
        while hfdp > self.alpha and np.sum(sc_rec * test_id) > 0:

            X_sc = X_aug[sc_rec <= 0,:]
            Y_sc = Y_aug[sc_rec <= 0]
            X_osc = X_aug[(sc_rec <= 0) * (sel_id >= 1),:] 
            Y_osc = Y_aug[(sc_rec <= 0) * (sel_id >= 1)]
            X_nsc = X_aug[(sc_rec <= 0) * (sel_id <= 0),:] ## newly screened data points
            Y_nsc = Y_aug[(sc_rec <= 0) * (sel_id <= 0)] ## newly screened data points
            X_usc = X_aug[sc_rec > 0,:] ## unscreened data points

            if(count % batch_size == 0): 
                if np.mean(1.0*(Y_train>0)) == 0:
                    Yhat_test = np.zeros(np.concatenate([X_calib, X_test]).shape[0])
                elif np.mean(1.0*(Y_train>0)) == 1:
                    Yhat_test = np.ones(np.concatenate([X_calib, X_test]).shape[0])
                else:
                    if mdl_select:
                        self.model_selection(X_train, Y_train, X_osc, Y_osc, X_nsc, Y_nsc, X_usc, [])
                    self.model_update(X_train, Y_train, X_sc, Y_sc, X_usc)

                    Yhat_test = self.model.predict_proba(np.concatenate([X_calib, X_test]))[:,1]

                    # if self.self_train:
                    #     Yhat_test = self.model.predict_proba(np.concatenate([X_calib, X_test]))[:,1]
                    # else:
                    #     Yhat_test = self.model.predict(np.concatenate([X_calib, X_test]))
                
                max_range = np.max(Yhat_test) + 100
                Yhat_test[sc_rec==0] = max_range * 1.
                if div_index > 0:
                    usc_mem = membership_mat[sc_rec > 0,:] ## membership matrix of the screened calibration data points
                    group_size = np.sum(usc_mem, axis = 0)
                    endangered_group = (group_size < div_index)
                    endangered_id = np.sum(usc_mem[:,endangered_group], axis = 1)
                    Yhat_test[sc_rec>0][endangered_id>0] = max_range * 1.
                    
                sel_id = np.ones(n+m)
            
            next_id = np.random.choice(np.where(Yhat_test == np.min(Yhat_test))[0])
            sc_rec[next_id] = 0
            Yhat_test[next_id] = max_range * 1.
            count +=1
            sel_id[next_id] = 0
           
            if div_index > 0:
                usc_mem = membership_mat[sc_rec > 0,:] ## membership matrix of the screened calibration data points
                group_size = np.sum(usc_mem, axis = 0)
                endangered_group = (group_size < div_index)
                endangered_id = np.sum(usc_mem[:,endangered_group], axis = 1)
                Yhat_test[sc_rec>0][endangered_id>0] = max_range * 1.
           
            Y_all = np.concatenate([Y_calib, np.zeros(m)])
            if np.sum(sc_rec * test_id) == 0:
                break
            hfdp = (1 + np.sum(sc_rec * cal_id * (Y_all <= self.c))) / np.sum(sc_rec * test_id) * m / (1+n)

        return (sc_rec * test_id)[n:] 


    def seqstep(self, Y_calib, Yhat):
       hfdp = 1
       n = len(Y_calib) 
       m = len(Yhat) - n
       sc_rec = np.ones(n+m)
       cal_id = np.ones(n+m)
       cal_id[n:] = 0
       test_id = np.zeros(n+m)
       test_id[n:] = 1
       max_range = np.max(Yhat) + 100

       while hfdp > self.alpha and np.sum(sc_rec * test_id) > 0:
           next_id = my_argmin(Yhat)
           # next_id = np.random.choice(np.where(Yhat == np.min(Yhat))[0])
           sc_rec[next_id] = 0
           Yhat[next_id] = max_range
           if np.sum(sc_rec * test_id) == 0:
               break
           Y_all = np.concatenate([Y_calib, np.zeros(m)])
           hfdp = (1 + np.sum(sc_rec * cal_id * (Y_all <= self.c))) / np.sum(sc_rec * test_id) * m / (1+n)
        
       return (sc_rec * test_id)[n:] 



    def compute_es(self, X_remain, pred, type = 'kernel', sig = 1):
        n = X_remain.shape[0]
        es = np.zeros((n,n)) 
        if type == 'kernel':
            for i in range(n):
                new_es = np.exp(-np.sum((X_remain[i,:] - X_remain)**2/sig**2, axis = 1)) * pred * pred[i]
                es[i,:] = new_es 
        if type == 'cosine':
            for i in range(n):
                new_es = np.dot(X_remain[i,:], X_remain.T) / np.linalg.norm(X_remain[i,:]) / np.linalg.norm(X_remain.T) * pred * pred[i]
                es[i,:] = new_es
        return es



def BH(calib_scores, test_scores, q = 0.1):
    ntest = len(test_scores)
    ncalib = len(calib_scores)
    pvals = np.zeros(ntest)
    
    for j in range(ntest):
        pvals[j] = (np.sum(calib_scores < test_scores[j]) + np.random.uniform(size=1)[0] * (np.sum(calib_scores == test_scores[j]) + 1)) / (ncalib+1)
         
    
    # BH(q) 
    df_test = pd.DataFrame({"id": range(ntest), "pval": pvals}).sort_values(by='pval')
    
    df_test['threshold'] = q * np.linspace(1, ntest, num=ntest) / ntest 
    idx_smaller = [j for j in range(ntest) if df_test.iloc[j,1] <= df_test.iloc[j,2]]
    
    if len(idx_smaller) == 0:
        return(np.array([]))
    else:
        idx_sel = np.array(df_test.index[range(np.max(idx_smaller)+1)])
        return(idx_sel)

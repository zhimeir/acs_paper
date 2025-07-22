import numpy as np
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR, SVC
from sklearn.linear_model import LogisticRegression, LinearRegression    
from sklearn.model_selection import KFold

class acs:
    def __init__(self, alpha, c, mdl_list = ['SVR', 'GB', 'RF', 'LR'], type = "class", div_index = 0, div_sig = 1):
        
        self.alpha = alpha ## target FDR level
        self.c = c ## threshold for the positive label
        self.mdl_list = mdl_list
        self.regressor = mdl_list[0] ## default regressor
        self.div_index = div_index ## diversity index
        self.div_sig = div_sig
        self.type = type ## "class" or "regress"


    def filter(self, X_train, Y_train, X_calib, Y_calib, X_test, batch_size = None, prescreen = None):
        
        """ configurations """
        n = len(X_calib)
        m = len(X_test)
        mdl_select = False
        if len(self.mdl_list) > 1: ## if there are multiple models to choose from
            mdl_select = True 
        if batch_size is None:
            batch_size = int(np.ceil(n/20))

        """ initialization """
        sc_rec = np.ones(n+m) # record the screening results: 1 for not screened, 0 for screened
        if prescreen is None: # determine whether to prescreen the positive calibration data points
            X_remain_init = np.concatenate([X_calib[Y_calib <= self.c], X_test])
            X_aug_init = np.concatenate([X_train,X_calib[Y_calib > self.c]])
            Y_aug_init = np.concatenate([Y_train,Y_calib[Y_calib > self.c],])
            kfold_init = KFold(n_splits=5, shuffle=True, random_state=42)
            sel, acc = self.cv_model_select(X_train, Y_train, X_remain_init, self.regressor, kfold_init)
            sel_pre, acc_pre = self.cv_model_select(X_aug_init, Y_aug_init, X_remain_init, self.regressor, kfold_init)
            if max(sel, sel_pre) > 0:
                prescreen = [True, False][np.argmax([sel_pre, sel])]    
            else:
                prescreen = [True, False][np.argmin([acc_pre, acc])]
        self.prescreen = prescreen
        if prescreen: 
            sc_rec[np.where(Y_calib > self.c)[0]] = 0 # screen the calibration data points with positive labels
        cal_id = np.ones(n+m)
        cal_id[n:] = 0
        test_id = np.zeros(n+m)
        test_id[n:] = 1
        count = 0 # keep track of the batch size
        Y_all = np.concatenate([Y_calib, np.zeros(m)])
        
        ## initialization of hfdp
        hfdp = (1 + np.sum(sc_rec * cal_id * (Y_all <= self.c))) / np.sum(sc_rec * test_id) * m / (1+n)
        #self.record = []     
        #self.sclist = []     
        self.models = []
        self.esrec = []
        
        """ sequantial screening """
        while hfdp > self.alpha and np.sum(sc_rec * test_id) > 0: ## proceed if the hfdp is larger than the threshold and there are unscreened test data points

            #sc_cal = sc_rec * cal_id
            sc_cal = sc_rec[:n] 
            X_sc = X_calib[sc_cal<=0,:] ## screened calibration data points
            Y_sc = Y_calib[sc_cal<=0]
            X_usc = X_calib[sc_cal>0,:]  ## unscreened calibration data points
            X_remain = np.concatenate([X_usc, X_test]) ## unscreened calibration and ALL test data points

            if(count % batch_size == 0): ## update the model every batch_size iterations 
                if mdl_select: ## select the best model
                    self.model_selection(X_train, Y_train, X_sc, Y_sc, X_remain)
                self.model_update(X_train, Y_train, X_sc, Y_sc, X_remain)
                count += 1

                Yhat_test = self.model.predict(np.concatenate([X_calib, X_test])) 
                """ diagnostics """ 
                self.models.append(self.model)
                #max_range = np.max(Yhat_test) + 100
                #Yhat_test[sc_rec==0] = max_range * 1.


                if self.div_index > 0: ## diversity regularization 
                    X_usc = np.concatenate([X_calib, X_test])[sc_rec == 1,:]
                    Yhat_usc = Yhat_test[sc_rec == 1] 
                    self.es = self.compute_es(X_usc, Yhat_usc, sig = self.div_sig)
                    es_inv = np.linalg.inv(self.es)
                    base1 = np.dot(es_inv, Yhat_usc)
                    base2 = np.dot(es_inv, np.ones(len(Yhat_usc)))
                    AA = np.dot(Yhat_usc,base2)
                    BB = np.dot(Yhat_usc,base1)
                    CC = np.dot(np.ones(len(Yhat_usc)),base2)
                    div_prob = (-AA + BB / (1-self.alpha)) * base2 - (AA / 1-self.alpha - CC) * base1
                    Yhat_test[sc_rec == 1] = (1 -  self.div_index) * Yhat_usc + self.div_index * div_prob / np.max(div_prob)
                    self.esrec.append((self.es.sum() - len(Yhat_usc)) / len(Yhat_usc) / (len(Yhat_usc) - 1))

            
            usc_id = np.where(sc_rec > 0)[0] ## indices of unscreened units
            Yhat_usc = Yhat_test[usc_id]
            next_id0 = np.random.choice(np.where(Yhat_usc == np.min(Yhat_usc))[0]) ## randomly choose one of the minimum values if there are ties
            next_id = usc_id[next_id0] ## get the index of the selected data point in the original data set 
            sc_rec[next_id] = 0
            #Yhat_test[next_id] = max_range * 1.

           
            if cal_id[next_id] == 1:
                count +=1
            
            if np.sum(sc_rec * test_id) == 0:
                break
            hfdp = (1 + np.sum(sc_rec * cal_id * (Y_all <= self.c))) / np.sum(sc_rec * test_id) * m / (1+n)


        return (sc_rec * test_id)[n:] 

    def model_update(self, X_train, Y_train, X_sc, Y_sc, X_remain):

        X_train_aug = np.concatenate((X_train, X_sc))
        Y_train_aug = np.concatenate((Y_train, Y_sc))

        if self.regressor == 'SVR':
            self.model = SVR(kernel="rbf", gamma=0.1)

        if self.regressor == 'GB': 
            self.model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=0)

        if self.regressor == 'RF':
            self.model = RandomForestRegressor(max_depth=5, random_state=0)
        
        if self.regressor == 'LR':
            self.model = LinearRegression()

        """ update the model """
        if self.type == 'class': ## ind(Y>c) ~ X 
            self.model.fit(X_train_aug, 1.*(Y_train_aug>self.c))
        if self.type == 'regress': ## Y ~ X
            self.model.fit(X_train_aug, Y_train_aug)


    def model_selection(self, X_train, Y_train, X_sc, Y_sc, X_remain): 
        
        X_train_aug = np.concatenate((X_train, X_sc))
        Y_train_aug = np.concatenate((Y_train, Y_sc))
        sel_list = [] # record the metrics of different models
        acc_list = []
        kfold = KFold(n_splits=5, shuffle=True, random_state=42)
        for method in self.mdl_list:
            
            sel, accuracy = self.cv_model_select(X_train_aug, Y_train_aug, X_remain, method, kfold)
            sel_list.append(sel)
            acc_list.append(accuracy)
            #method_res = 0
            #for train_index, test_index in kfold.split(X_train_aug):
            #    X_train_k, X_test_k = X_train_aug[train_index,:], X_train_aug[test_index,:]
            #    Y_train_k, Y_test_k = Y_train_aug[train_index], Y_train_aug[test_index]
            #   
            #    if method == 'SVR':
            #        model = SVR(kernel="rbf", gamma=0.1)
            #
            #    if method == 'GB':
            #        model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=0)
            #    
            #    
            #    if method == 'RF':
            #        model = RandomForestRegressor(max_depth=5, random_state=0)
            #
            #    if self.type == 'class': 
            #        model.fit(X_train_k, 1.*(Y_train_k>self.c))
            #    if self.type == 'regress':
            #        model.fit(X_train_k, Y_train_k)
            #
            #    Yhat = model.predict(np.concatenate([X_test_k, X_remain]))
            #    
            #    if len(X_test_k) > 0:
            #        method_res += np.sum(self.seqstep(Y_test_k, Yhat))
            #    else: 
            #        method_res += - np.mean((1*(Y_train_aug>self.c) - model.predict(X_train_aug))**2)
            
            #res_list.append(np.mean(method_res))

        if max(sel_list) > 0:
            winner = np.argmax(sel_list)
        else:
            winner = np.argmin(acc_list)
        self.regressor = self.mdl_list[winner]


    
    """ when screened test data points can be labeled """    
    def filter_aug(self, X_train, Y_train, X_calib, Y_calib, X_test, Y_test, batch_size = None):
        
        """ configurations """
        n = len(X_calib)
        m = len(X_test)
        X_aug = np.concatenate([X_calib, X_test])
        Y_aug = np.concatenate([Y_calib, Y_test])
        mdl_select = False
        if len(self.mdl_list) > 1: # if there are multiple models to choose from
            mdl_select = True 
        if batch_size is None:
            batch_size = int(np.ceil(n/20))
       
        """ initialization """
        sc_rec = np.ones(n+m) # record the screening results: 1 for not screened, 0 for screened
        cal_id = np.ones(n+m)
        cal_id[n:] = 0
        test_id = np.zeros(n+m)
        test_id[n:] = 1
        count = 0 # keep track of the batch size 
        Y_all = np.concatenate([Y_calib, np.zeros(m)])
        
        ## initialization of hfdp
        hfdp = (1 + np.sum(sc_rec * cal_id * (Y_all <= self.c))) / np.sum(sc_rec * test_id) * m / (1+n)
        self.models = []     

        """ sequantial screening """
        while hfdp > self.alpha and np.sum(sc_rec * test_id) > 0:

            X_sc = X_aug[sc_rec <= 0,:]
            Y_sc = Y_aug[sc_rec <= 0]
            X_usc = X_aug[sc_rec > 0,:] ## unscreened data points

            if(count % batch_size == 0): 
                if mdl_select:
                    self.model_selection(X_train, Y_train, X_sc, Y_sc, X_usc)
                self.model_update(X_train, Y_train, X_sc, Y_sc, X_usc)
                
                Yhat_test = self.model.predict(np.concatenate([X_calib, X_test]))
                #self.record.append(Yhat_test*1.)
                #self.sclist.append(sc_rec*1)
                self.models.append(self.model)
                #max_range = np.max(Yhat_test) + 100
                #Yhat_test[sc_rec==0] = max_range * 1.
                    
                #sel_id = np.ones(n+m)
            
            usc_id = np.where(sc_rec > 0)[0] ## indices of unscreened units
            Yhat_usc = Yhat_test[usc_id]
            next_id0 = np.random.choice(np.where(Yhat_usc == np.min(Yhat_usc))[0]) ## randomly choose one of the minimum values if there are ties
            next_id = usc_id[next_id0] ## get the index of the selected data point in the original data set 
            sc_rec[next_id] = 0
            count +=1
           
            if np.sum(sc_rec * test_id) == 0:
                break
            hfdp = (1 + np.sum(sc_rec * cal_id * (Y_aug <= self.c))) / np.sum(sc_rec * test_id) * m / (1+n)

        return (sc_rec * test_id)[n:] 


    def seqstep(self, Y_calib, Yhat):
       
       n = len(Y_calib) 
       m = len(Yhat) - n
       sc_rec = np.ones(n+m)
       cal_id = np.ones(n+m)
       cal_id[n:] = 0
       test_id = np.zeros(n+m)
       test_id[n:] = 1
       max_range = np.max(Yhat) + 100
       Y_all = np.concatenate([Y_calib, np.zeros(m)])
       hfdp = (1 + np.sum(sc_rec * cal_id * (Y_all <= self.c))) / np.sum(sc_rec * test_id) * m / (1+n)
       
       while hfdp > self.alpha and np.sum(sc_rec * test_id) > 0:
           #next_id = np.argmin(Yhat)
           next_id = np.random.choice(np.where(Yhat == np.min(Yhat))[0])
           sc_rec[next_id] = 0
           Yhat[next_id] = max_range * 1.
           if np.sum(sc_rec * test_id) == 0:
               break
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


    def cv_model_select(self, X_train_aug, Y_train_aug, X_remain, mdl, kfold):

        sel_res = 0
        accuracy = 0
        for train_index, test_index in kfold.split(X_train_aug):
            
            X_train_k, X_test_k = X_train_aug[train_index,:], X_train_aug[test_index,:]
            Y_train_k, Y_test_k = Y_train_aug[train_index], Y_train_aug[test_index]

            if mdl == 'SVR':
                model = SVR(kernel="rbf", gamma=0.1)
            if mdl == 'GB':
                model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=0)
            if mdl == 'RF':
                model = RandomForestRegressor(max_depth=5, random_state=0)
            if mdl == 'LR':
                if self.type == 'class':
                    model = LogisticRegression()
                if self.type == 'regress':
                    model = LinearRegression()

            if self.type == 'class':
                model.fit(X_train_k, 1.*(Y_train_k>self.c))
            if self.type == 'regress':
                model.fit(X_train_k, Y_train_k)
            
            Yhat = model.predict(np.concatenate([X_test_k, X_remain]))
            accuracy += np.sum((1.*(Y_test_k>self.c) - 1.*(Yhat[:len(X_test_k)]>self.c))**2)
            sel_res += np.sum(self.seqstep(Y_test_k, Yhat))
        
        sel_res = sel_res / kfold.n_splits
        accuracy = accuracy / kfold.n_splits
        
        return sel_res, accuracy
    



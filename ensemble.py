import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from deslib.des.knora_u import KNORAU
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

class Ensemble_classifier:
    """
    This class takes predictor variables and outcome variable,
    gives f1 score of the ensemble learning based classifier
    Parameters
    ----------
    X: array  
      predictor variables
    
    y: array
      outcome variable
     
    """

    def __init__(self,X,y,test_size,n_estimators,scale_pos_weight,sampling_strategy,
                 oversampling_strategy,num_iter):
        self.X=X
        self.y=y
        self.test_size=test_size
        self.n_estimators=n_estimators
        self.scale_pos_weight=scale_pos_weight
        self.sampling_strategy=sampling_strategy
        self.oversampling_strategy=oversampling_strategy
        self.num_iter=num_iter
        self.random_state=42
        self.n_jobs=-1
        self.num_neighbour=10
        
    # getting training and test samples
    def get_samples(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
           self.X, self.y,test_size=self.test_size, random_state=self.random_state)
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
     # define the pool of classifiers    
    def get_classifiers(self):
        self.pool= [RandomForestClassifier(n_estimators=self.n_estimators,
                                           random_state=self.random_state,
                                           n_jobs=self.n_jobs),
              BaggingClassifier(n_estimators=self.n_estimators,random_state=
                                self.random_state, n_jobs=self.n_jobs)]
        return self.pool
    
    # generating new samples via over-sampling and under-samplings algorithms
    def get_new_samples(self):
        
        under_sampler = SMOTE(sampling_strategy=self.sampling_strategy,
                              random_state=self.random_state)
        over_sampler=RandomUnderSampler(sampling_strategy=self.oversampling_strategy)
        self.X_new, self.y_new =under_sampler.fit_resample(self.X_train, 
                                                                 self.y_train)
        self.X_new, self.y_new =over_sampler.fit_resample(self.X_new, self.y_new)
        return  self.X_new, self.y_new
    
    # getting the f1 score
    def get_result(self):
        pool=self.get_classifiers()
        self.results = list()
        for i in range(self.num_iter):
            #get new samples
            self.X_train, self.X_test, self.y_train, self.y_test=self.get_samples()
            self.X_new, self.y_new=self.get_new_samples()
            # building the pool of classifiers
            for p in pool:
                p.fit(self.X_new, self.y_new)
            model = KNORAU(pool_classifiers=self.pool,k=self.num_neighbour)
            # fit the model
            model.fit(self.X_new, self.y_new)
            # getting f1-scores
            yhat = model.predict(self.X_test)
            score= f1_score(self.y_test, yhat)
            self.results.append(score)
        print('>%s %.3f (%.3f)' % ('Ensemble', np.mean(self.results)
                                   , np.std(self.results)))
            
        return self.results

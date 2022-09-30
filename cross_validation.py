import numpy as np
import pandas as pd
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import RepeatedStratifiedKFold,cross_val_score

class Cross_validation():
    """
    This class takes predictor variables and outcome variable
    and a list of machine learning models, and 
    gives f1 score of the machine learning models
    Parameters
    ----------
    X: array  
      predictor variables
    
    y: array
      outcome variable
      
    models: list
      list of models
      
    names: list
      list of models' names
    """
    # class variables
    random_state=1
    n_jobs=-1
    
    def __init__(self,X,y,models,names,n_splits,n_repeats,vif_cutoff):
        self.X=X
        self.y=y
        self.models=models
        self.names=names
        self.n_splits=n_splits
        self.n_repeats=n_repeats
        self.vif_cutoff=vif_cutoff
    
    def vif_truncator(self):
        # Calculating VIF
        X_dataframe=pd.DataFrame(self.X)
        vif = pd.DataFrame()
        vif["variables"] = X_dataframe.columns
        vif["VIF"] = [variance_inflation_factor(X_dataframe.values, i) 
                      for i in range(X_dataframe.shape[1])]
        # removing rows where their vif values are larger than a threshold from X
        v_list=list(vif[vif.VIF > self.vif_cutoff].variables)
        self.X=np.delete(self.X, [i for i in v_list], 1)
        return(self.X)
    
    # evaluate a model
    def evaluate_model(self, model,random_state):
        # define evaluation procedure
        cv = RepeatedStratifiedKFold(n_splits=self.n_splits, n_repeats=self.n_repeats,
                                     random_state=random_state)
        # evaluate model 
        scores = cross_val_score(model, self.X, self.y,n_jobs=self.n_jobs, 
                                 scoring='f1', cv=cv)
        return scores
    
    # getting the f1 score
    def cross_scores(self,random_state=random_state):
        self.results = list()
        # evaluate each model
        for i in range(len(self.models)):
         if self.names[i]=='LR':
            self.X=self.vif_truncator()
            self.scores=self.evaluate_model(self.models[i],random_state)
            self.results.append(self.scores)
            print('>%s %.3f (%.3f)' % (self.names[i], np.mean(self.scores)
                                       , np.std(self.scores)))
         else:
            self.scores=self.evaluate_model(self.models[i],random_state)
            self.results.append(self.scores)
            print('>%s %.3f (%.3f)' % (self.names[i], np.mean(self.scores)
                                       , np.std(self.scores)))
        return self.results
    
    
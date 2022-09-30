from sklearn.ensemble import RandomForestClassifier,BaggingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

class Classifiers():
    
    def get_bagging (self):
        classifier=BaggingClassifier(n_estimators = 100,random_state = 42, 
                                        n_jobs = -1)
        name='Bagging'
        return classifier,name
    
    def get_randomforest (self):
        classifier=RandomForestClassifier(n_estimators = 100, random_state = 42, 
                                             n_jobs = -1)
        name='RF'
        return classifier,name
    
    def get_knn (self):
        classifier=KNeighborsClassifier()
        name='KNN'
        return classifier,name
    
    def get_LR (self):
        classifier=LogisticRegression(solver='sag')
        name='LR'
        return classifier,name
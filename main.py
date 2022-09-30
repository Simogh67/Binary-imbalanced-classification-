import pandas as pd
from data_processing import Data_processing
import matplotlib.pyplot as plt
from cross_validation import Cross_validation
from classifiers import Classifiers
from ensemble import Ensemble_classifier
from cnn import CNN

# PARAMETERS ================
# cross_validation parameters
n_splits=5
n_repeats=3
vif_cutoff=2.6
# Ensemble learning parameters
test_size=0.2
n_estimators=100
scale_pos_weight=5
sampling_strategy=0.4 # rate of under-sampling strategy
oversampling_strategy=0.8 # rate of over-sampling strategy
num_iter=5 # number of the validation set approach iterations
#CNN parameters 
cutoff=.75
num_first_filter=128
num_second_filter=64
num_third_filter=32
drop_rate=0.5
kernel_size=3
test_size=0.2
pool_size=2
epochs=100
batch_size=100
num_iter=5 #number of the validation set approach iterations
weights= {0:1, 1:10} # weights for cost-sensitive learning

# Functions ================

def read_data(file):
    df=pd.read_csv(file,sep=';')
    return df

    
def get_models():
    models, names = list(), list()
    # Bagging
    item=Classifiers()
    model,name=item.get_bagging()
    models.append(model)
    names.append(name)
    # Random Forest
    model,name=item.get_randomforest()
    models.append(model)
    names.append(name)
    # KNN
    model,name=item.get_knn()
    models.append(model)
    names.append(name)
    # LR
    model,name=item.get_LR()
    models.append(model)
    names.append(name)
    
    return models, names


def main():
    # reading data file
    df=read_data('bank.csv') 
    # getting customers
    item=Data_processing(df)
    X,y=item.get_customers()
    # define models
    models,names=get_models()
    # reporting f1 scores
    item=Cross_validation(X,y,models,names,n_splits,n_repeats,vif_cutoff)
    results=item.cross_scores()
    # ensembling results
    item=Ensemble_classifier(X,y,test_size,n_estimators,scale_pos_weight,
                             sampling_strategy,oversampling_strategy,num_iter)
    result=item.get_result()
    results.append(result)
    names.append('Ensemble')
    #CNN results
    item=CNN(X,y,test_size,cutoff,num_first_filter,num_second_filter,
                  num_third_filter,kernel_size, drop_rate,pool_size,num_iter,
                  epochs,batch_size,weights)
    result=item.get_result()
    results.append(result)
    names.append('CNN')
    print(results)
    # plot the results
    plt.boxplot(results, labels=names, showmeans=True)
    plt.title('f1_score')
    plt.show()


if __name__ == "__main__":
    main()
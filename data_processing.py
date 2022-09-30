from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,Normalizer
import numpy as np

class Data_processing():
    """
    This class takes the dataframe, and gives predictor variables and 
    outcome variable required to train machine learning models
    Parameters
    ----------
    df: dataframe 
      contains customers' information
    """
    def __init__(self,df):
        self.df=df
        
    def get_customers(self):
        """This function removes unnecessary columns and anomalies 
           from the data. Also it generates the response and predictor 
           variables. 
        """
        # removing unnecessary columns and anomalies
        self.df=self.df[self.df['previous']==0]
        self.df=self.df[self.df['campaign']<=30]
        self.df=self.df[self.df['balance']<=35000]
        self.df = self.df.drop(columns=['contact','day','default',
                                            'pdays','previous','poutcome'],axis=1)
        
        self.df.drop(self.df.index[self.df['job']=='unknown'],
             inplace=True)
        self.df.drop(self.df.index[self.df['education']=='unknown'], 
             inplace=True)
        # seperating column default from the dataset
        self.y=self.df['y']
        self.df=self.df.drop(columns='y')
        
        # select categorical and numerical features+ one-hot encosing+ normalized the numerical columns
        cat_ind = self.df.select_dtypes(include=['object']).columns
        num_ind = self.df.select_dtypes(include=['int64']).columns
        ct = ColumnTransformer([('o',OneHotEncoder(drop='first'),cat_ind)
                                ,('n',Normalizer(),num_ind)])
        
        # the matrix contains features
        self.X = (ct.fit_transform(self.df))
        # making the binary targets 
        self.y=self.y.to_numpy()
        self.y=np.where(self.y == 'no', 0, 1)
        self.X=self.X.toarray()
        return self.X, self.y
    

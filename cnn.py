import numpy as np
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

class CNN:
    """
    This class takes predictor variables with a group of the neural network  
    parameters, and the outcome variable, gives f1 score of the convolutional 
    neural network classifier
    Parameters
    ----------
    X: array  
      predictor variables
    
    y: array
      outcome variable
     
    """
    
    def __init__(self,X,y,test_size,cutoff,num_first_filter,num_second_filter,
                 num_third_filter,kernel_size, drop_rate,pool_size,num_iter,
                 epochs,batch_size,weights):
        self.X=X
        self.y=y
        self.cutoff=cutoff
        self.num_first_filter=num_first_filter
        self.num_second_filter=num_second_filter
        self.num_third_filter=num_third_filter
        self.drop_rate=drop_rate
        self.kernel_size=kernel_size
        self.test_size=test_size
        self.pool_size=pool_size
        self.epochs=epochs
        self.batch_size=batch_size
        self.num_iter=num_iter
        self.weights=weights 
        self.random_state=42
        
    #define the model    
    def get_model(self,n_features):
        model = Sequential()
        model.add(Conv1D(filters=self.num_first_filter, kernel_size=self.kernel_size,
                         activation='elu',input_shape=(n_features,1)))
        model.add(MaxPooling1D(pool_size=self.pool_size))
        model.add(Dropout(self.drop_rate))
        model.add(Conv1D(filters=self.num_second_filter,kernel_size=self.kernel_size,
                         activation='elu'))
        model.add(MaxPooling1D(pool_size=self.pool_size))
        model.add(Dropout(self.drop_rate))
        model.add(Conv1D(filters=self.num_third_filter, kernel_size=self.kernel_size,
                         activation='elu'))
        model.add(MaxPooling1D(pool_size=self.pool_size))
        model.add(Flatten())
        model.add(Dense(8,  activation='elu', kernel_initializer='he_uniform'))
        model.add(Dense(1, activation='sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='adam', 
                      metrics=['accuracy'])
        return model
    
    # getting training and test samples
    def get_samples(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size,random_state=self.random_state)
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    # getting the f1 score
    def get_result(self):
        self.results= list()
        for i in range(self.num_iter):
            self.X_train, self.X_test, self.y_train, self.y_test=self.get_samples()
            model =self.get_model(self.X_train.shape[1])
        # fit the model
            model.fit(self.X_train, self.y_train, self.epochs, self.batch_size,
                      class_weight=self.weights)
            yhat_classes = (model.predict(self.X_test) > self.cutoff).astype("int32")
            score = f1_score(self.y_test, yhat_classes)
            self.results.append(score)
        print('>%s %.3f (%.3f)' % ('CNN', np.mean(self.results)
                                   , np.std(self.results)))
        return self.results
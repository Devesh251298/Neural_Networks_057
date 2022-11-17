import torch
import pickle
import numpy as np
import pandas as pd
from torch.autograd import Variable
from sklearn.impute import SimpleImputer
pd.options.mode.chained_assignment = None
class Regressor():

    def __init__(self, x, nb_epoch = 1000):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """ 
        Initialise the model.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape 
                (batch_size, input_size), used to compute the size 
                of the network.
            - nb_epoch {int} -- number of epochs to train the network.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Replace this code with your own
        X, _ = self._preprocessor(x, training = True)
        self.input_size = X.shape[1]
        self.output_size = 1
        self.nb_epoch = nb_epoch 
        self.median_train_dict=dict() # Stores all median values for training data.
        self.learning_rate = 0.0001
        self.linear_model = torch.nn.Linear(self.input_size, self.output_size)
        self.mse_loss = torch.nn.MSELoss()
        self.optimiser = torch.optim.SGD(self.linear_model.parameters(), lr = self.learning_rate) 

        return 

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


    def _preprocessor(self, x, y = None, training = False):
            """ 
            Preprocess input of the network.

            Arguments:
                - x {pd.DataFrame} -- Raw input array of shape 
                    (batch_size, input_size).
                - y {pd.DataFrame} -- Raw target array of shape (batch_size, 1).
                - training {boolean} -- Boolean indicating if we are training or 
                    testing the model.

            Returns:
                - {torch.tensor} or {numpy.ndarray} -- Preprocessed input array of
                  size (batch_size, input_size). The input_size does not have to be the same as the input_size for x above.
                - {torch.tensor} or {numpy.ndarray} -- Preprocessed target array of
                  size (batch_size, 1).

            """
            #######################################################################
            #                       ** START OF YOUR CODE **
            #######################################################################

            # Replace this code with your own
            # Return preprocessed x and y, return None for y if it was None


            if training==True:
                # Replace all x Nan with median
                x_columns=x.select_dtypes(include=['float']).columns
                self.median_train_dict=dict()

                def replace_NA(df,column_name):
                    median_col=df[column_name].median()
                    self.median_train_dict[column_name]=median_col
                    df[column_name].fillna(median_col,inplace=True)

                    return df.isnull().sum()


                for column in x_columns:
                    replace_NA(x,column)
                    
                # Replace all x 'object' types Nan with median
                # imp = SimpleImputer(missing_values='NAN', strategy='most_frequent', fill_value=None)
                # new_col=imp.fit_transform(np.array(x['ocean_proximity']).reshape(-1, 1))

                # x['ocean_proximity']=new_col
                most_freq_cat=x['ocean_proximity'].value_counts()[0]#.to_frame()
                self.median_train_dict['ocean_proximity']=most_freq_cat
          
                x['ocean_proximity'].fillna(most_freq_cat,inplace=True)

                
                # Replace all y Nan with median
                if y is None:
                    pass
                else:
                    y_columns=y.columns
                    median_col_y=y["median_house_value"].median()
                    self.median_train_dict["median_house_value"]=median_col_y
                    y["median_house_value"].fillna(median_col_y,inplace=True)
                    y=y.to_numpy()

                #Convert all the clean data to numerical data
                one_hot_encoded_data=pd.get_dummies(x, columns = ['ocean_proximity'])
                
                #Normalize all the clean data
                x=(one_hot_encoded_data-one_hot_encoded_data.min())/(one_hot_encoded_data.max()-one_hot_encoded_data.min())


                
            else: #if data is test data
                
                x_columns=x.select_dtypes(include=['float']).columns
                # Replace all x Nan with median
                def replace_NA(df,column_name):
                    df[column_name].fillna(self.median_train_dict[column_name],inplace=True)

                    return df.isnull().sum()


                for column in x_columns:
                    replace_NA(x,column)
                
                # Replace all x 'object' types Nan with median
                x["ocean_proximity"].fillna(self.median_train_dict['ocean_proximity'],inplace=True)
                if y is None:
                    pass
                else:
                    
                    y_columns=y.columns
                    y["median_house_value"].fillna(self.median_train_dict["median_house_value"],inplace=True)
                    y=y.to_numpy()
                
                #Convert all the clean data to numerical data
                one_hot_encoded_data=pd.get_dummies(x, columns = ['ocean_proximity'])
                
                #Normalize all the clean data
                x=(one_hot_encoded_data-one_hot_encoded_data.min())/(one_hot_encoded_data.max()-one_hot_encoded_data.min())         
                

            x=x.to_numpy()
            
            return x, (y if isinstance(y, (np.ndarray, np.generic,pd.DataFrame)) else None)

            #######################################################################
            #                       ** END OF YOUR CODE **
            #######################################################################

        
    def fit(self, x, y):
        """
        Regressor training function

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            self {Regressor} -- Trained model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################
     
        X, Y = self._preprocessor(x, y = y, training = True) # Do not forget
        
        # Transform numpy arrays into tensors
        X = Variable(torch.from_numpy(X).type(dtype=torch.FloatTensor))
        Y = Variable(torch.from_numpy(Y).type(dtype=torch.FloatTensor))
        for epoch in range(self.nb_epoch):
            
            # Compute a forward pass
            output = self.linear_model(X) 

            # Compute MSE based on forward pass
            loss_forward = self.mse_loss(output, Y)
          
            # Perform backward pass 
            loss_forward.backward()

            # Update parameters of the model
            self.optimiser.step()

            print('epoch {}'.format(epoch))

        return self

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

            
    def predict(self, x):
        """
        Output the value corresponding to an input x.

        Arguments:
            x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).

        Returns:
            {np.ndarray} -- Predicted value for the given input (batch_size, 1).

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, _ = self._preprocessor(x, training = False) # Do not forget
        pass

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def score(self, x, y):
        """
        Function to evaluate the model accuracy on a validation dataset.

        Arguments:
            - x {pd.DataFrame} -- Raw input array of shape 
                (batch_size, input_size).
            - y {pd.DataFrame} -- Raw output array of shape (batch_size, 1).

        Returns:
            {float} -- Quantification of the efficiency of the model.

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        X, Y = self._preprocessor(x, y = y, training = False) # Do not forget
        return 0 # Replace this code with your own

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def save_regressor(trained_model): 
    """ 
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    with open('part2_model.pickle', 'wb') as target:
        pickle.dump(trained_model, target)
    print("\nSaved model in part2_model.pickle\n")


def load_regressor(): 
    """ 
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    with open('part2_model.pickle', 'rb') as target:
        trained_model = pickle.load(target)
    print("\nLoaded model in part2_model.pickle\n")
    return trained_model



def RegressorHyperParameterSearch(): 
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented 
    in the Regressor class.

    Arguments:
        Add whatever inputs you need.
        
    Returns:
        The function should return your optimised hyper-parameters. 

    """

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################

    return  # Return the chosen hyper parameters

    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################



def example_main():

    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas DataFrame as inputs
    data = pd.read_csv("housing.csv") 

    # Splitting input and output
    x_train = data.loc[:, data.columns != output_label]
    y_train = data.loc[:, [output_label]]

    # Training
    # This example trains on the whole available dataset. 
    # You probably want to separate some held-out data 
    # to make sure the model isn't overfitting
    regressor = Regressor(x_train, nb_epoch = 10)


    regressor.fit(x_train, y_train)
    save_regressor(regressor)

    # Error
    error = regressor.score(x_train, y_train)
    print("\nRegressor error: {}\n".format(error))


if __name__ == "__main__":
    example_main()

    #Testing preprocessor
    # x_pre_proc,y_pre_proc=regressor._preprocessor(x_train,y_train,training=True)
    # print(f' x type = {type(x_pre_proc)},y type = {type(y_pre_proc)}')


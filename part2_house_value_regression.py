import torch
import pickle
import numpy as np
import pandas as pd
from torch.autograd import Variable
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from torch import nn
import torch
import traceback



pd.options.mode.chained_assignment = None
class Regressor():

    def __init__(self, x, neurons=[1], activations=["relu"], learning_rate = 0.001, batch_size = 64, nb_epoch = 1000):
        # You can add any input parameters you need
        # Remember to set them with a default value for LabTS tests
        """ 
        Initialise the model.
          
        Arguments:
            - x {pd.DataFrame} -- Raw input data of shape 
                (batch_size, input_size), used to compute the size 
                of the network.
            - neurons {list[int]} -- list specifying the number of
                neurons in each linear layer (length = depth of the nn)
            - activations {list[str]} -- list containing the activation
                functions after each linear layer
            - learning_rate {float} -- learning rate to train the network
            - nb_epoch {int} -- number of epochs to train the network.
            - batch_size {int} -- number of instances in each batch

        """

        #######################################################################
        #                       ** START OF YOUR CODE **
        #######################################################################

        # Replace this code with your own
        try:
            X, _ = self._preprocessor(x, training = True)
            print(f"Constructing a regressor with X of shape: {X.shape}")

            self.x = x
            self.median_train_dict=dict() # Stores all median values for training data.
            self.input_size = X.shape[1]
            print(f"setting input size: {self.input_size}")
            self.output_size = 1
            self.nb_epoch = nb_epoch 
            self.learning_rate = learning_rate
            self.batch_size = batch_size
            self.neurons = neurons
            self.activations = activations
            self.one_hot_encoded_data_train_min=0
            self.one_hot_encoded_data_train_max=0
            self.cat_dummies=list() #stores all training dummy variables
            self.processed_columns=list() #stores all training columns including dummy variables

            # build layers specified by self.neurons and self.activations
            self._layers = []
            for layer_num in range(len(self.neurons)):
                # add linear layer
                n_out = self.neurons[layer_num]
                if layer_num == 0:
                    n_in = self.input_size
                else:
                    n_in = self.neurons[layer_num-1]
                print(f"input size: {self.input_size}")
                print(f"Building linear layer of shape: {n_in},{n_out}")
                self._layers.append(nn.Linear(n_in, n_out))
                
                # add activation function
                if self.activations[layer_num] == "relu":
                    self._layers.append(nn.ReLU())
                else:
                    self._layers.append(nn.Sigmoid())
                
            # build torch neural network with the specified layers
            self.model = nn.Sequential(*self._layers)
            
            # initialize weights randomly (otherwise they stay at zero)
            self.model.apply(self._weights_init)
            
            # define loss and optimimser
            self.mse_loss = torch.nn.MSELoss()
            self.optimiser = torch.optim.SGD(self.model.parameters(), lr = self.learning_rate) 
        except Exception:
            traceback.print_exc()

        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################

    def get_params(self, deep=False):
        """ Method to get parameters needed to perform sklearn grid search."""
        hyperparameters = {
            'x': self.x,
            'neurons': self.neurons,
            'activations': self.activations,
            'batch_size': self.batch_size,
            'nb_epoch': self.nb_epoch,
            'learning_rate': self.learning_rate
        }
        return hyperparameters
    
    def set_params(self, **parameters):
        """ Method to set parameters needed to perform sklearn grid search."""
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
    
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

            try :
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

                    #Storing one_hot encoded data to be used in validation set
                    cat_columns = ["ocean_proximity"]
                
                    #create the dummy variables
                    df_processed = pd.get_dummies(x,prefix_sep="__",columns=cat_columns)

                    #store the dummy variables from training dataset
                    self.cat_dummies = [col for col in df_processed if "__" in col and col.split("__")[0] in cat_columns]        
                    #store all columns of the training dataset in a list
                    self.processed_columns = list(df_processed.columns[:])  

                    #Normalize all the clean data
                    self.one_hot_encoded_data_train_min=one_hot_encoded_data.min()
                    self.one_hot_encoded_data_train_max=one_hot_encoded_data.max()
                    x=(one_hot_encoded_data-self.one_hot_encoded_data_train_min)/(self.one_hot_encoded_data_train_max-self.one_hot_encoded_data_train_min)


                    
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
                    # one_hot_encoded_data=pd.get_dummies(x, columns = ['ocean_proximity'])
                    
                    #one hot encode the test data set
                    cat_columns = ["ocean_proximity"]

                    df_test_processed = pd.get_dummies(x, prefix_sep="__", columns=cat_columns)

                    #removing additional features from test set
                    for col in df_test_processed.columns:
                        if ("__" in col) and (col.split("__")[0] in cat_columns) and col not in self.cat_dummies:
                            df_test_processed.drop(col, axis=1, inplace=True)  

                    #adding 0's to missing features in test set
                    for col in self.cat_dummies:
                        if col not in df_test_processed.columns:
                            df_test_processed[col] = 0

                    # print(self.processed_columns)

                    #add all missing One hot encoders
                    df_test_processed = df_test_processed[self.processed_columns]
                    one_hot_encoded_data=df_test_processed
                        
                    #Normalize all the clean data
                    x=(one_hot_encoded_data-one_hot_encoded_data.min())/(one_hot_encoded_data.max()-one_hot_encoded_data.min())         
                    
                x=x.to_numpy()
                
                return x, (y if isinstance(y, (np.ndarray, np.generic,pd.DataFrame)) else None)

            except Exception:
                traceback.print_exc()
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
        try : 
            header_template = "Training a Regressor model with: x shape: {}, y shape: {}, neurons: {}, activations: {}, lr: {}, batch-size: {}, nb_epochs: {}"
            print(header_template.format(x.shape, y.shape, self.neurons, self.activations, self.learning_rate, self.batch_size, self.nb_epoch))
            X, Y = self._preprocessor(x, y = y, training = True) # Do not forget
            # Transform numpy arrays into tensors
            X = Variable(torch.from_numpy(X).type(dtype=torch.FloatTensor)).requires_grad_(True)
            Y = Variable(torch.from_numpy(Y).type(dtype=torch.FloatTensor)).requires_grad_(True)

            for epoch in range(self.nb_epoch):
                train_loss = 0
                avg_y_pred = 0
                samples = 0
                self.model.train()

                for (batch_X, batch_Y) in self._next_batch(X, Y, self.batch_size):
                    # Zero the gradients
                    self.optimiser.zero_grad()
                    # Compute a forward pass
                    output = self.model(batch_X) 
                    
                    # add the sum of y_pred to compute the avg
                    avg_y_pred += output.sum().item()
                    
                    # Compute MSE based on forward pass
                    loss_forward = self.mse_loss(output, batch_Y)
                
                    # Perform backward pass 
                    loss_forward.backward()

                    # Update parameters of the model
                    self.optimiser.step()
                    train_loss += loss_forward.item() * batch_Y.size(0)
                    samples += batch_Y.size(0)
                if epoch%10 == 0:
                    print("Epoch: {}...".format(epoch+1))
                    train_template = "epoch: {} train rmse: {:e} avg y pred: {:e}"
                    print(train_template.format(epoch + 1, (np.sqrt(train_loss) / samples),
                        avg_y_pred / samples))

        except Exception:
            traceback.print_exc()
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

        # X, _ = self._preprocessor(x, training = False) # Do not forget
        try : 
            X, _ = self._preprocessor(x, training = False)
            print(f"Predicting Y for X of shape {X.shape}")

            X = Variable(torch.from_numpy(X).type(dtype=torch.float32 ))
            output = self.model(X)
            return output.detach().numpy()

        except Exception:
            traceback.print_exc()
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
        try :
            X, Y = self._preprocessor(x, y, training = False) # Do not forget
            print(f"Scoring with x of shape {X.shape} and y of shape {Y.shape}")

            y_model = self.predict(x)
            return mean_squared_error(Y, y_model, squared=False) # Replace this code with your own
 
        except Exception:
            traceback.print_exc()
        
        
    def _next_batch(self, inputs, targets, batchSize):
        try:
            # loop over the dataset
            for i in range(0, inputs.shape[0], batchSize):
                # yield a tuple of the current batched data and labels
                yield (inputs[i:i + batchSize], targets[i:i + batchSize])
        except Exception:
            traceback.print_exc()

    def _weights_init(self, m):
        try:
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                torch.nn.init.zeros_(m.bias)
        except Exception:
            traceback.print_exc()
        #######################################################################
        #                       ** END OF YOUR CODE **
        #######################################################################


def save_regressor(trained_model): 
    """ 
    Utility function to save the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with load_regressor
    try : 
        with open('part2_model.pickle', 'wb') as target:
            pickle.dump(trained_model, target)
        print("\nSaved model in part2_model.pickle\n")
    except Exception:
        traceback.print_exc()


def load_regressor(): 
    """ 
    Utility function to load the trained regressor model in part2_model.pickle.
    """
    # If you alter this, make sure it works in tandem with save_regressor
    try : 
        with open('part2_model.pickle', 'rb') as target:
            trained_model = pickle.load(target)
        print("\nLoaded model in part2_model.pickle\n")
        return trained_model
    
    except Exception:
        traceback.print_exc()



def RegressorHyperParameterSearch(x_train, y_train): 
    # Ensure to add whatever inputs you deem necessary to this function
    """
    Performs a hyper-parameter for fine-tuning the regressor implemented 
    in the Regressor class.
    
    We will perform hyper-parameter tuning using a bottom-up approach.
    First we will fine-tune a NN with 1 hidden layer, and get the optimal
    number of neurons. Then we will fine-tune a NN with 2 hidden layers,
    fixing the number of neurons for the first layer and tuning the number
    of neurons of the second neuron. Etcetera...

    Arguments:
        Add whatever inputs you need.
        
    Returns:
        The function should return your optimised hyper-parameters. 

    """

    #######################################################################
    #                       ** START OF YOUR CODE **
    #######################################################################
    
    # do hyperparam tuning of a nn with 1 hidden layer
    n_out = 1
    n_in = x_train.shape[1]
    # rule of thumb: number of hidden neurons should be less than twice input size
    n_hidden = list(range(n_out, 2*n_in))
    
    # define grid to search with 5-fold cross-validation
    hyperparameters = {
        'x': [x_train],
        'neurons': [[num_neurons,1] for num_neurons in n_hidden],
        'activations': [["relu","relu"], ["sigmoid","relu"]],
        'batch_size': [64, 128, 256],
        'nb_epoch': [10],
        'learning_rate': [0.1, 0.01, 0.001]
    }
    # define grid search on the Regressor estimator with 5-fold cross-validation
    gs = GridSearchCV(Regressor(x_train), hyperparameters, cv=5)
    
    # fit the training data
    gs.fit(x_train,  y_train)
    breakpoint()
    # do hyperparam tuning of a nn with 2 hidden layers
    return  # Return the chosen hyper parameters


    #######################################################################
    #                       ** END OF YOUR CODE **
    #######################################################################



def hyperparam_main():

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
    
    # trying NN with 1 hidden layer with 18 (=2xinput dim) neurons
    """
    neurons = [1]
    activations = ["relu"]
    
    regressor = Regressor(x_train, neurons, activations, batch_size = 128, nb_epoch = 1000)

    regressor.fit(x_train, y_train)
    save_regressor(regressor)
    """
    
    # do hyperparam tuning
    RegressorHyperParameterSearch(x_train, y_train)

    # Error
    print(x_train.shape)
    error = regressor.score(x_train, y_train)
    print("\nRegressor error: {}\n".format(error))



def example_main():

    output_label = "median_house_value"

    # Use pandas to read CSV data as it contains various object types
    # Feel free to use another CSV reader tool
    # But remember that LabTS tests take Pandas DataFrame as inputs
    data = pd.read_csv("housing.csv") 

    # Splitting input and output
    x_train = data.loc[:, data.columns != output_label]
    y_train = data.loc[:, [output_label]]

    # x_train = data.loc[20:50, data.columns != output_label]
    # y_train = data.loc[20:50, [output_label]]



    # Training
    # This example trains on the whole available dataset. 
    # You probably want to separate some held-out data 
    # to make sure the model isn't overfitting
    regressor = Regressor(x_train, nb_epoch = 1000)


    x_pre_proc,y_pre_proc=regressor._preprocessor(x_train,y_train,training=True)
 



    #Uncomment this when done ############



    regressor.fit(x_train, y_train)
    save_regressor(regressor)

    # Error
    print(x_train.shape)
    error = regressor.score(x_train, y_train)
    print("\nRegressor error: {}\n".format(error))


if __name__ == "__main__":
    example_main()
    # hyperparam_main()
    #Testing preprocessor
    # x_pre_proc,y_pre_proc=regressor._preprocessor(x_train,y_train,training=True)
    # print(f' x type = {type(x_pre_proc)},y type = {type(y_pre_proc)}')
    # print(f' x type = {type(x_pre_proc)},y type = {type(y_pre_proc)}')
    # print(x_pre_proc)

    # x_test = data.loc[:10, data.columns != output_label]
    # y_test = data.loc[:10, [output_label]]

    # x_pre_proc_test,y_pre_proc_test=regressor._preprocessor(x_test,y_test,training=False)

    # print(f' length of training data={len(x_pre_proc.columns)}')
    # print(f' length of test data={len(x_pre_proc_test.columns)}')



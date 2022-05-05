# Tensorflow and Keras are two packages for creating neural network models.
import json
import pandas as pd # data from for the data.
import tensorflow as tf
from tensorflow import keras

# import NN layers and other componenets.
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, BatchNormalization, Dropout
from tensorflow.keras import optimizers

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt # for plotting data and creating different charts.
import numpy as np # for math and arrays
import seaborn as sns # for plotting.
import datetime

from DataHandling.DataImporter import DataImporter

def main():

    tf.random.set_seed(13) # to make sure the experiment is reproducible.
    tf.debugging.set_log_device_placement(False) #WAS MACHT DAS?

    all_ds = pd.read_csv('iris_dataset.csv')

    all_ds = all_ds.sample(frac=1) # This will randomly shuffle the rows to make sure the data is not sorted

    # Split the data into 60% train and 40% test (later will divide the test to test and validate.)
    train_dataset, temp_test_dataset =  train_test_split(all_ds, test_size=0.2)
    # Split the test_dataset dataframe to 50% test and 50% validation. [this will divide the dataset into 60% train, 20% validate, and 20% test]
    test_dataset, valid_dataset =  train_test_split(temp_test_dataset, test_size=0.5)

    train_labels1 = train_dataset.pop('target')
    test_labels1 = test_dataset.pop('target')
    valid_labels1 = valid_dataset.pop('target')

    # Encode the labeles
    train_labels = pd.get_dummies(train_labels1, prefix='Label')
    valid_labels = pd.get_dummies(valid_labels1, prefix='Label')
    test_labels = pd.get_dummies(test_labels1, prefix='Label')



    # Plot the relationship between each two variables to spot anything incorrect.
    train_stats = train_dataset.describe()
    sns.pairplot(train_stats[train_stats.columns], diag_kind="kde") # or diag_kind='reg'

    # Statistics on the train dataset to make sure it is in a good shape. (you may display the same stat for test and validate)
    train_stats = train_dataset.describe()
    #train_stats.pop("target")
    train_stats = train_stats.transpose()

    normed_train_data = pd.DataFrame(StandardScaler().fit_transform(train_dataset), columns=train_dataset.columns, index=train_dataset.index)
    normed_test_data = pd.DataFrame(StandardScaler().fit_transform(test_dataset), columns=test_dataset.columns, index=test_dataset.index)
    normed_valid_data = pd.DataFrame(StandardScaler().fit_transform(valid_dataset), columns=valid_dataset.columns, index=valid_dataset.index)
    
    print(normed_train_data.shape)
    print((normed_train_data.shape[1],))

    #%%time

    # We decalred a function for creating a model.
    def build_model1_two_hidden_layers():
        model = Sequential()
        model.add(Dense(16, input_shape = (normed_train_data.shape[1],)))    # Input layer => input_shape must be explicitly designated       
        #model.add(Dense(16,Activation('relu'))) # Hidden layer 1 => only output dimension should be designated (output dimension = # of Neurons = 50)
        model.add(Dense(3, activation='softmax'))                          # Output layer => output dimension = 1 since it is a regression problem
        # output neurons must be == number of possible classes
        # Activation: sigmoid, softmax, tanh, relu, LeakyReLU. 
        #Optimizer: SGD, Adam, RMSProp, etc. # https://www.tensorflow.org/api_docs/python/tf/keras/optimizers
        learning_rate = 0.0001
        optimizer = optimizers.Adam(learning_rate)
        # was macht compile()????????
        model.compile(loss='categorical_crossentropy',#from_logits=True),
                    optimizer=optimizer,
                    metrics=['accuracy']) # for regression problems, mean squared error (MSE) is often employed
        return model


    params = json.load(open(file="./params.json"))
    EPOCHS = params['EPOCHS']
    batch_size = params['batch_size'] #6 iteration

    # log_dir = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)


    model = build_model1_two_hidden_layers()
    print('Here is a summary of this model: ')
    model.summary()

    with tf.device('/CPU:0'): # it can be with '/CPU:0'
    # with tf.device('/GPU:0'): # comment the previous line and uncomment this line to train with a GPU, if available.
        history = model.fit(
            normed_train_data, 
            train_labels,
            batch_size = batch_size,
            epochs=EPOCHS, 
            verbose=1,
            shuffle=True,
            steps_per_epoch = int(normed_train_data.shape[0] / batch_size) ,
            validation_data = (normed_valid_data, valid_labels), 
            callbacks=[tensorboard_callback]  
        )

    print("run successfull")



if __name__ == '__main__':
    main()
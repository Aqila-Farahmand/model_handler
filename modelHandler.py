from abc import ABC, abstractmethod
from functools import reduce
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import os
from sklearn.preprocessing import LabelBinarizer
from config import SafetyCageConfig
from sklearn.metrics import f1_score

import pandas as pd

from typing import List, Optional

#Required to load CIFAR10 dataset in some cases (needed for mac but not windows)
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

def read_train_data(filename):

    train_data = pd.read_csv(filename)

    return train_data


def load_trained_model(filename):
    trained_model = keras.saving.load_model(filename, custom_objects=None, compile=True, safe_mode=True)
    
    return trained_model


class ModelHandler():
    """Class that handles keras model """
    def __init__(self, model_filename:str="MNIST_model", cfg:SafetyCageConfig=None):
       
        use_pre_activ_val = cfg.train.use_pre_activ_val
        num_classes = cfg.model.num_classes
        use_onehotencoder = cfg.model.use_onehotencoder
        selected_layers_str = cfg.train.selected_layers_str
        dataset_str = cfg.model.dataset
        use_rgb2grey = cfg.model.use_rgb2grey
        neuron_plot_dict = cfg.model.neuron_plot_dict
        self.selected_layers_list = cfg.train.selected_layers_list

        self.model = load_trained_model(model_filename)
        self.model_layers = self.model.layers
        self.model_shape = self.calc_model_shape(selected_layers_str)  # index 0 is the first hidden layer. inputlayer cannot be acessed in this way
        self.model_input = self.model.input
        self.classes = range(num_classes)
        self.use_rgb2grey = use_rgb2grey
        self.input_dim = self.model.input_shape[1:][0]
        
        self.x_train_correct:list = None
        self.y_train_binary_correct:list = None
        self.y_train_predicted:list = None
        self.y_train_pred_wrong:list = None 
        self.y_train_pred_ok:list = None
        self.y_test_pred_wrong:list = None
        self.y_test_pred_ok:list = None
        self.x_test_wrong:list = None
        self.x_test_ok:list = None
        self.correctly_classified_bool_list_train:list = None 
        self.correctly_classified_indices_boolean_test:list = None 

        self.y_predmodel:list = None
        self.y_test:list = None

        self.selected_layers_list = self.calc_selected_layers_list(selected_layers_str) if (self.selected_layers_list == []) else self.selected_layers_list
        
        if use_pre_activ_val == True:
            
            # We collect all the ingredients to compute the pre-activation values from the previous-layer activation.
            # The weights and biases are Keras-variables that contain the arrays for the weights and biases (numpy arrays).
            # The inputs are Keras tensors, do not contain values but are symbolic objects used in the computation when the Keras function is called.
            self.layer_inputs = [layer.input for layer in self.model.layers]
            self.layer_weights = [layer.kernel for layer in self.model.layers]
            self.layer_biases = [layer.bias for layer in self.model.layers]

            # This creates the Keras function that links the model's inputs to the individual layers inputs.
            # Note that the first layer's input is the same as the model's input, while the second and third layer's inputs will be the first and second layer's OUTputs, respectively.
            # That's why I call the arrays "previous layers activation values".
            self.pseudo_activation_function = K.function(self.model_input, self.layer_inputs) 
        else: 
            outputs = [layer.output for layer in self.model.layers]
            self.activation_function = K.function(self.model_input, outputs)


        dataset_prep_func_dict = {"MNIST":self.get_MNIST_data(), "CIFAR10":self.get_CIFAR10_data()}
        # Get correct data given dataset using a dictionary of functions
        ((x_train, y_train), (x_test, y_test)) = dataset_prep_func_dict[dataset_str]

        #NB! Here a small edit is done for NLDL revision. Remove and refacturate code afterwards:
        # There should be an ABC model_handler that not only handles the model, but also data used
        #to train safetycage. Tailored Modelhandlers should be made by based on this (in the same spirit as custommodelhandler was)
        #Train safetycage on half of test data, and evaluate on the other half. Do random splitting.
        #lb_test = LabelBinarizer()
        #y_test = lb_test.fit_transform(y_test)
        #x_train,x_test,y_train,y_test = train_test_split(x_test,y_test,test_size = 0.5)
        #y_test = np.argmax(y_test,axis = 1)

        #Store predictions:   
        self.y_train_predicted = np.argmax(self.model.predict(x_train), axis=1)

        
        if use_onehotencoder:
            self.correctly_classified_bool_list_train:list[bool] = self.y_train_predicted == np.argmax(y_train, axis=1)
        else:
            self.correctly_classified_bool_list_train:list[bool] = self.y_train_predicted == y_train

        #Correctly classified vs wrongly classified
        

        self.x_train_correct = x_train[self.correctly_classified_bool_list_train]
        self.x_train_wrong = x_train[~self.correctly_classified_bool_list_train]

        self.y_train_binary_correct = y_train[self.correctly_classified_bool_list_train]
        self.y_train_binary_wrong = y_train[~self.correctly_classified_bool_list_train]


        self.correctly_classified_indices_boolean_train = self.y_train_predicted == (np.argmax(y_train,axis = 1))


        # To be used for testing safetycage
        # model:
        self.y_test_predicted = np.argmax(self.model.predict(x_test), axis=1)

        self.initial_model_accuracy = f1_score(y_test, self.y_test_predicted, average="macro")

        self.correctly_classified_indices_boolean_test = self.y_test_predicted == y_test
        correct_indices = np.where(self.correctly_classified_indices_boolean_test)[0]

        # Separate wrongly classified from corectly classified
        # model 1:
        self.y_train_pred_ok = self.y_train_predicted[self.correctly_classified_indices_boolean_train]
        self.y_train_pred_wrong = self.y_train_predicted[~self.correctly_classified_indices_boolean_train]
        
        self.y_test_pred_ok = self.y_test_predicted[self.correctly_classified_indices_boolean_test]
        self.x_test_ok = x_test[self.correctly_classified_indices_boolean_test, :]
        self.y_test_pred_wrong = self.y_test_predicted[~self.correctly_classified_indices_boolean_test]
        self.x_test_wrong = x_test[~self.correctly_classified_indices_boolean_test, :]

        self.y_predmodel = np.concatenate(
        (self.y_test_pred_wrong, self.y_test_pred_ok))

        self.y_pred_train = np.concatenate(
        (self.y_train_pred_wrong, self.y_train_pred_ok))

        self.y_train = np.concatenate(
        (y_train[~self.correctly_classified_indices_boolean_train], y_train[self.correctly_classified_indices_boolean_train]))

        self.x_train = np.concatenate(
        (x_train[~self.correctly_classified_indices_boolean_train], x_train[self.correctly_classified_indices_boolean_train]))

        self.x_test = np.concatenate(
        (x_test[~self.correctly_classified_indices_boolean_test], x_test[self.correctly_classified_indices_boolean_test])) 

        self.y_test = np.concatenate(
        (y_test[~self.correctly_classified_indices_boolean_test], y_test[self.correctly_classified_indices_boolean_test]))    

    def get_MNIST_data(self):
        ((raw_x_train, raw_y_train), (raw_x_test, raw_y_test)) = keras.datasets.mnist.load_data()

        x_train = raw_x_train.reshape((raw_x_train.shape[0], 28 * 28 * 1))
        x_test = raw_x_test.reshape((raw_x_test.shape[0], 28 * 28 * 1))
        # scale data to the range of [0, 1]
        x_train = x_train.astype("float32") / 255.0
        x_test = x_test.astype("float32") / 255.0

        lb = LabelBinarizer()
        y_train = lb.fit_transform(raw_y_train)

        return ((x_train, y_train), (x_test, raw_y_test))

    def get_CIFAR10_data(self):
        ((raw_x_train, raw_y_train), (raw_x_test, raw_y_test)) = keras.datasets.cifar10.load_data()

        # Flatten y_train and y_test to get in same shape as for MNIST data:
        y_train_flat = raw_y_train.flatten()
        y_test_flat = raw_y_test.flatten()

        if self.use_rgb2grey == False:
            x_train = raw_x_train.reshape((raw_x_train.shape[0], 32*32*3))
            x_test = raw_x_test.reshape((raw_x_test.shape[0], 32*32*3))
            # scale data to the range of [0, 1]
            x_train = x_train.astype("float32") / 255.0
            x_test = x_test.astype("float32") / 255.0

        else:
            def rgb2gray(rgb):
                return np.dot(rgb[..., :3], [0.299, 0.587, 0.144])

            # Convert rgb to gray-scale images:
            x_train = np.array([rgb2gray(i) for i in raw_x_train])
            x_test = np.array([rgb2gray(i) for i in raw_x_test])
            # Flatten images to vectors:
            x_train = x_train.reshape((x_train.shape[0], 32*32*1))
            x_test = x_test.reshape((x_test.shape[0], 32*32*1))
            # scale data to the range of [0, 1]
            x_train = x_train.astype("float32") / 255.0
            x_test = x_test.astype("float32") / 255.0

        lb = LabelBinarizer()
        y_train = lb.fit_transform(y_train_flat)
        y_test = y_test_flat

        return ((x_train, y_train), (x_test, y_test))


    def calc_pre_activations(self, x)-> list[np.float64]: #list of floats # Uses keras model to get pre-activ. values for an input x

        previous_layer_activ_vals = self.pseudo_activation_function(x)

        # Then, from the previous layers activation, finding the raw (i.e. pre-activation) inputs to the layers is a simple matrix-vector product.
        # pre_activation_input_to_layer_i = output_of_layer_i-1 * weights + bias.
        pre_activ_values = [np.dot(inps, w.numpy()) + b.numpy() for inps, w, b in zip(
            previous_layer_activ_vals, self.layer_weights, self.layer_biases)]

        # Rename to keep compatibility with the rest of the code below
        return pre_activ_values

        # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    
    def calc_activations(self, x) -> list[np.float64]: #Mainly just uses the given activation function
        return self.activation_function(x)
    

    def calc_model_shape(self, selected_layers_str:str):
        temp_shape = [None]*len(self.model.layers)
        for i in range(len(self.model.layers)):
            if type(self.model.layers[i].output.shape) == tuple:
                temp_shape[i] = reduce(lambda x, y: x * y,
                                 self.model.layers[i].output.shape[1:])
            elif type(self.model.layers[i].output.shape) == list:
                temp_shape[i] = reduce(lambda x, y: x * y,
                                 self.model.layers[i].output_shape[0][1:])


        return temp_shape
    

    def calc_selected_layers_list(self, selected_layers_str:str):
        #prepare selected layers (all vs hidden vs output etc) -legacy version is layers_used
        all_layers_range_list = list(range(len(self.model_shape)))
        if selected_layers_str == "all":
            selected_layers_list = all_layers_range_list
        elif selected_layers_str == "output":
            selected_layers_list = [all_layers_range_list[-1]]
        elif selected_layers_str == "penultimate":
            selected_layers_list = [all_layers_range_list[-2]]
        elif selected_layers_str == "hidden":
            selected_layers_list = all_layers_range_list[0:-1]
        
        return selected_layers_list
    
    def plot_neuron_distributions(self, onehotencoder:bool, use_pre_activ_val:bool, layers_neuron_values, neuron_plot_dict:dict, y):
        # plot distributions of activation

        # Which layers to plot activations:
        plot_layers = neuron_plot_dict["layers"]

        # Which particular correctly predicted class to plot from
        plot_class = neuron_plot_dict["class"]

        if plot_layers == "all":
            layers_to_plot = layers_neuron_values
        elif plot_layers == "output":
            layers_to_plot = layers_neuron_values[len(layers_neuron_values)-1]
        elif plot_layers == "penultimate":
            layers_to_plot = [
                layers_neuron_values[len(layers_neuron_values)-2]]
        elif plot_layers == "hidden":
            layers_to_plot = layers_neuron_values[0:-1]

        # Which nodes to plot for each layer. This should be a list of lists, with length corresponding to number of layers chosen to plot from:
        nodes_per_layer = neuron_plot_dict["nodes"]

        for l in range(len(layers_to_plot)):  # for all layers

            # The activations for layer i for the class chosen
            if onehotencoder:
                plot_class_activation = layers_to_plot[l][y[:,
                                                            plot_class] == 1, :]

            else:
                plot_class_activation = layers_to_plot[l][y ==
                                                            plot_class, :]
            num_histograms = len(nodes_per_layer[l])
            num_columns = num_histograms
            rows = int(np.ceil(num_histograms / num_columns))
            figsize_width = 5  # You can adjust this based on your preference
            figsize_height = figsize_width * (rows / num_columns)

            fig, axs = plt.subplots(1, len(nodes_per_layer[l]), figsize=(
                figsize_width, figsize_height))
            if rows == 1 or num_columns == 1:
                axs = np.ravel(axs)
            else:
                axs = axs.flatten()

            # The colors of each neuron
            colors = ["blue", "orange"]

            # for all neurons chosen in the layer
            for index, n in enumerate(nodes_per_layer[l]):
                axs[index].hist(plot_class_activation[:, n],
                                bins=100, color=colors[index])
                #axs[index].set_title('Layer ' + str(i) + ' - Neuron ' + str(n))
                #fig.suptitle('Class ' + str(k))
            if use_pre_activ_val == True:
                fig.suptitle("Pre-activation values", fontsize=16)
                fig.tight_layout()
                plt.savefig("histogram_pre_activation.png")
            else:
                fig.suptitle("Activation values", fontsize=16)
                fig.tight_layout()
                plt.savefig("histogram_activation.png")
            
            plt.show(block=False)

"""
    If you have a Neural Network that is not a feed foward NN or is emplemented in Keras or you want handle your model or calculate activation values/  
    equivalent intermediate output values in a diffent way use the Custom Model Handler. intermediate output values must be 
    calculated either in "calc_activations(self, x) or calc_pre_activations(self, x)" and return a #List[np.array[np.float64]] with  
    len of list is layers in model and shape of arrays is (number of samples in x,nodes in layer)
"""
class CustomModelHandler():
    #init
        #Load model, get raw training and testing data, predict with model 
    def __init__(self, model_filename:str="MNIST_model", cfg:SafetyCageConfig=None):
       
        use_pre_activ_val = cfg.train.use_pre_activ_val
        num_classes = cfg.model.num_classes
        use_onehotencoder = cfg.model.use_onehotencoder
        #selected_layers_str = cfg.train.selected_layers_str
        dataset_str = cfg.model.dataset
        use_rgb2grey = cfg.model.use_rgb2grey
        neuron_plot_dict = cfg.model.neuron_plot_dict
        self.selected_layers_list = cfg.train.selected_layers_list
        
        
        self.model = "Load your model here" #load_trained_model(os.path.join(os.getcwd(), model_filename)) 
    

        self.model_input = "Change if you are not using Keras" #self.model.input 

        self.x_train_correct:list = None
        self.y_train_binary_correct:list = None
        self.y_train_predicted:list = None 
        self.y_test_pred_wrong:list = None
        self.y_test_pred_ok:list = None
        self.x_test_wrong:list = None
        self.x_test_ok:list = None
        self.correctly_classified_bool_list_train:list = None 
        self.correctly_classified_indices_boolean_test:list = None 

        self.y_predmodel:list = None
        self.y_test:list = None

        self.model_shape = "must be an array with length equal to the number layers in the model and values equal to the number of nodes in each layer" #self.calc_model_shape(selected_layers_str)  

        self.selected_layers_list = self.selected_layers_list
        
        ###Modify this code to suit your needs, If you are not using a KERAS model
        ###Keep in mind that in the current implementation we gather activation valse 
        ###For all layers, if you want to change that you must change how these values are accessed in the SafetyCage class  
        if use_pre_activ_val == True:
            
            # We collect all the ingredients to compute the pre-activation values from the previous-layer activation.
            # The weights and biases are Keras-variables that contain the arrays for the weights and biases (numpy arrays).
            # The inputs are Keras tensors, do not contain values but are symbolic objects used in the computation when the Keras function is called.
            self.layer_inputs = [layer.input for layer in self.model.layers]
            self.layer_weights = [layer.kernel for layer in self.model.layers]
            self.layer_biases = [layer.bias for layer in self.model.layers]

            # This creates the Keras function that links the model's inputs to the individual layers inputs.
            # Note that the first layer's input is the same as the model's input, while the second and third layer's inputs will be the first and second layer's OUTputs, respectively.
            # That's why I call the arrays "previous layers activation values".
            self.pseudo_activation_function = K.function(self.model_input, self.layer_inputs) 
        else: 
            outputs = [layer.output for layer in self.model.layers]
            self.activation_function = K.function(self.model_input, outputs)


        dataset_prep_func_dict = {"MNIST":self.get_MNIST_data(), "CIFAR10":self.get_CIFAR10_data(), "YOURDATASET":self.get_YOUR_data()}
        # Get correct data given dataset using a dictionary of functions
        ((x_train, y_train), (x_test, y_test)) = dataset_prep_func_dict[dataset_str]


        #Store correctly classified values    
        self.y_train_predicted = np.argmax(self.model.predict(x_train), axis=1)

        
        if use_onehotencoder:
            self.correctly_classified_bool_list_train:list[bool] = self.y_train_predicted == np.argmax(y_train, axis=1)
        else:
            self.correctly_classified_bool_list_train:list[bool] = self.y_train_predicted == y_train

        #Correctly classified vs wrongly classified
        

        self.x_train_correct = x_train[self.correctly_classified_bool_list_train]
        self.x_train_wrong = x_train[~self.correctly_classified_bool_list_train]

        self.y_train_binary_correct = y_train[self.correctly_classified_bool_list_train]
        self.y_train_binary_wrong = y_train[~self.correctly_classified_bool_list_train]

        # To be used for testing safetycage
        # model:
        self.y_test_predicted = "We want the prediction labels here" #np.argmax(self.model.predict(x_test), axis=1)

        self.initial_model_accuracy = f1_score(y_test, self.y_test_predicted, average="macro")

        self.correctly_classified_indices_boolean_test = self.y_test_predicted == y_test
        correct_indices = np.where(self.correctly_classified_indices_boolean_test)[0]

        # Separate wrongly classified from corectly classified
        # model 1:
        self.y_test_pred_ok = self.y_test_predicted[self.correctly_classified_indices_boolean_test]
        self.x_test_ok = x_test[self.correctly_classified_indices_boolean_test, :]
        self.y_test_pred_wrong = self.y_test_predicted[~self.correctly_classified_indices_boolean_test]
        self.x_test_wrong = x_test[~self.correctly_classified_indices_boolean_test, :]

        self.y_predmodel = np.concatenate(
        (self.y_test_predicted[~self.correctly_classified_indices_boolean_test], self.y_test_predicted[self.correctly_classified_indices_boolean_test]))

        self.y_test = np.concatenate(
        (y_test[~self.correctly_classified_indices_boolean_test], y_test[self.correctly_classified_indices_boolean_test]))    

    def get_YOUR_data(self):

            # ((raw_x_train, raw_y_train), (raw_x_test, raw_y_test)) = keras.datasets.mnist.load_data()

            # x_train = raw_x_train.reshape((raw_x_train.shape[0], 28 * 28 * 1))
            # x_test = raw_x_test.reshape((raw_x_test.shape[0], 28 * 28 * 1))
            # # scale data to the range of [0, 1]
            # x_train = x_train.astype("float32") / 255.0
            # x_test = x_test.astype("float32") / 255.0

            # lb = LabelBinarizer()
            # y_train = lb.fit_transform(raw_y_train)

            # return ((x_train, y_train), (x_test, raw_y_test))
        pass

    def get_MNIST_data(self):
        ((raw_x_train, raw_y_train), (raw_x_test, raw_y_test)) = keras.datasets.mnist.load_data()

        x_train = raw_x_train.reshape((raw_x_train.shape[0], 28 * 28 * 1))
        x_test = raw_x_test.reshape((raw_x_test.shape[0], 28 * 28 * 1))
        # scale data to the range of [0, 1]
        x_train = x_train.astype("float32") / 255.0
        x_test = x_test.astype("float32") / 255.0

        lb = LabelBinarizer()
        y_train = lb.fit_transform(raw_y_train)

        return ((x_train, y_train), (x_test, raw_y_test))

    def get_CIFAR10_data(self):
        ((raw_x_train, raw_y_train), (raw_x_test, raw_y_test)) = keras.datasets.cifar10.load_data()

        # Flatten y_train and y_test to get in same shape as for MNIST data:
        y_train_flat = raw_y_train.flatten()
        y_test_flat = raw_y_test.flatten()

        if self.use_rgb2grey == False:
            x_train = raw_x_train.reshape((raw_x_train.shape[0], 32*32*3))
            x_test = raw_x_test.reshape((raw_x_test.shape[0], 32*32*3))
            # scale data to the range of [0, 1]
            x_train = x_train.astype("float32") / 255.0
            x_test = x_test.astype("float32") / 255.0

        else:
            def rgb2gray(rgb):
                return np.dot(rgb[..., :3], [0.299, 0.587, 0.144])

            # Convert rgb to gray-scale images:
            x_train = np.array([rgb2gray(i) for i in raw_x_train])
            x_test = np.array([rgb2gray(i) for i in raw_x_test])
            # Flatten images to vectors:
            x_train = x_train.reshape((x_train.shape[0], 32*32*1))
            x_test = x_test.reshape((x_test.shape[0], 32*32*1))
            # scale data to the range of [0, 1]
            x_train = x_train.astype("float32") / 255.0
            x_test = x_test.astype("float32") / 255.0

        lb = LabelBinarizer()
        y_train = lb.fit_transform(y_train_flat)
        y_test = y_test_flat

        return ((x_train, y_train), (x_test, y_test))

    ###Modify this code to suit your needs, If you are not using a KERAS model
    ###Keep in mind that in the current implementation we gather activation valse 
    ###For all layers, if you want to change that you must change how these values are accessed in the SafetyCage class  
        

    def calc_pre_activations(self, x) -> list[np.float64]: #array of floats # Uses keras model to get pre-activ. values for an input x

            previous_layer_activ_vals = self.pseudo_activation_function(x)

            # Then, from the previous layers activation, finding the raw (i.e. pre-activation) inputs to the layers is a simple matrix-vector product.
            # pre_activation_input_to_layer_i = output_of_layer_i-1 * weights + bias.
            pre_activ_values = [np.dot(inps, w.numpy()) + b.numpy() for inps, w, b in zip(
                previous_layer_activ_vals, self.layer_weights, self.layer_biases)]

            # Rename to keep compatibility with the rest of the code below
            return pre_activ_values

            # <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<

    ###Modify this code to suit your needs, If you are not using a KERAS model
    ###Keep in mind that in the current implementation we gather activation valse 
    ###For all layers, if you want to change that you must change how these values are accessed in the SafetyCage class  
        
    def calc_activations(self, x) -> list[np.float64]: #Mainly just uses the given activation function
        return self.activation_function(x) 
   
    pass

if __name__ == '__main__':
    #Quick test to confirm it builds
    model_handler = ModelHandler()

    print(len(model_handler.x_train_correct))
from typing import List, Any, Dict
from functools import reduce

import numpy as np
import keras.backend as K

from src.model.model_handler.model_handler import ModelHandler


class ModelHandlerKeras(ModelHandler):
    def __init__(
        self,
        selected_layers: str,
        use_pre_activation_values: bool,
        use_onehot_encoder: bool,
        model: Any,
    ):
        super(ModelHandlerKeras, self).__init__(
            selected_layers,
            use_pre_activation_values,
            use_onehot_encoder,
            model,
        )
        """
        Initialize ModelHandler with pre-loaded dataset and model.
        ModelHandler needs to calculate the following:
            - pre_activations of any given layer
            - activations of any given layer
            - model_shape:
                list that contains the total number of nodes/neurons in each layer of the neural network
            - selected_layers_list:
                specify which layers of the neural network should be analyzed.
        Args:
            dataset: Tuple of ((x_train, y_train), (x_test, y_test))
            model: Pre-trained model instance
            selected_layers: List of layer indices to analyze
        """

        self.model = model
        self.model_layers = self.model.layers
        self.model_input = self.model.input
        self.model_shape = self._calc_model_shape()
        
        if use_pre_activation_values:
            # We collect all the ingredients to compute the pre-activation values from the previous-layer activation.
            # The weights and biases are Keras-variables that contain the arrays for the weights and biases (numpy arrays).
            # The inputs are Keras tensors, do not contain values but are symbolic objects used in the computation when the Keras function is called.
            self.layer_inputs = [layer.input for layer in self.model_layers]
            self.layer_weights = [layer.kernel for layer in self.model_layers]
            self.layer_biases = [layer.bias for layer in self.model_layers]

            # This creates the Keras function that links the model's inputs to the individual layers inputs.
            # Note that the first layer's input is the same as the model's input, while the second and third layer's inputs will be the first and second layer's OUTputs, respectively.
            # That's why I call the arrays "previous layers activation values".
            self.pseudo_activation_function = K.function(self.model_input, self.layer_inputs) 
        else: 
            outputs = [layer.output for layer in self.model_layers]
            self.activation_function = K.function(self.model_input, outputs)

    def _get_predictions(self, x: np.ndarray) -> np.ndarray:
        """Get model predictions for input x."""
        
        raw_predictions = self.model.predict(x)
        return np.argmax(raw_predictions, axis=1)
    

    def _get_correct_predictions_mask(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Create boolean mask of correct predictions."""
        if len(y_true.shape) > 1:  # Handle one-hot encoded labels
            y_true = np.argmax(y_true, axis=1)
        return y_true == y_pred
    

    def _get_activations(self, x: np.ndarray) -> List[np.ndarray]:
        """Calculate activations for each layer given input x."""
        
        activations = {}
        
        for model_layer in self.model_layers:
            layer_output = model_layer.output
            get_activations = K.function([self.model.input], [layer_output])
            activations[model_layer.name] = get_activations([x])[0]

        return activations
    

    def _get_pre_activations(self, x: np.ndarray) -> List[np.ndarray]:
        """Calculate pre-activation values for each layer given input x."""
        
        self.layer_inputs = [layer.input for layer in self.model_layers]
        self.layer_weights = [layer.kernel for layer in self.model_layers]
        self.layer_biases = [layer.bias for layer in self.model_layers]
        
        previous_layer_activation = K.function(self.model_input, self.layer_inputs) 
        
        previous_layer_activation_values = previous_layer_activation(x)

        # Then, from the previous layers activation, finding the raw (i.e. pre-activation) inputs to the layers is a simple matrix-vector product.
        # pre_activation_input_to_layer_i = output_of_layer_i-1 * weights + bias.
        pre_activation_values = [
            np.dot(inps, w.numpy()) + b.numpy() for inps, w, b in zip(
                previous_layer_activation_values, self.layer_weights, self.layer_biases
                )
            ]
    # def _calc_model_shape(self) -> List[int]:
    #     """
    #     Calculate the shape of each layer in the model.
    #     This method iterates over the model layers and computes the shape of the output for each layer.
    #     The shape is calculated by multiplying the dimensions of the output shape, excluding the batch size.
        
    #     Returns:
    #         List[int]: A list containing the calculated shape for each layer in the model.
    #     """
        
    #     temp_shape = [None]*len(self.model_layers)
    #     for i in range(len(self.model_layers)):
    #         if type(self.model_layers[i].output.shape) == tuple:
    #             temp_shape[i] = reduce(
    #                 lambda x, y: x * y,
    #                 self.model_layers[i].output.shape[1:]
    #                 )
    #         elif type(self.model_layers[i].output.shape) == list:
    #             temp_shape[i] = reduce(
    #                 lambda x, y: x * y,
    #                 self.model_layers[i].output_shape[0][1:]
    #                 )
    #     return temp_shape
    
            
    def _calc_model_shape(self) -> Dict[str,int]:
        """
        Returns a dictionary mapping layer names to their output shapes.
        
        Parameters:
        model: Keras model
        
        Returns:
        dict: Keys are layer names, values are shape tuples
        """
        shape_dict = {}
        for layer in self.model.layers:
            shape_dict[layer.name] = layer.output_shape[1]
        return shape_dict
        
if __name__ == '__main__':
    model_handler = ModelHandlerKeras()
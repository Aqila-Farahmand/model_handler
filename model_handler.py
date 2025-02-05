from abc import ABC, abstractmethod
from typing import Tuple, Union, List, Any, Dict
import numpy as np

class ModelHandler(ABC):
    def __init__(
        self,
        selected_layers:str,
        use_pre_activation_values: bool,
        use_onehot_encoder:bool,
        model:Any
        ):
        super(ModelHandler, self).__init__()
        self.model = model
        
        
        self.use_pre_activation_values = use_pre_activation_values
        self.use_onehot_encoder = use_onehot_encoder
        self.selected_layers = selected_layers
        
        # self.selected_layers_list = self.calc_selected_layers_list(selected_layers_str) if (self.selected_layers_list == []) else self.selected_layers_list

    @abstractmethod
    def _get_predictions(self, x: np.ndarray) -> np.ndarray:
        """Get model predictions for input x."""
        raise NotImplementedError("Implement based on your model architecture")
    
    
    @abstractmethod
    def _get_correct_predictions_mask(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Create boolean mask of correct predictions."""
        raise NotImplementedError("Implement based on your model architecture")
    
    
    @abstractmethod
    def _get_activations(self, x: np.ndarray) -> List[np.ndarray]:
        """Calculate activations for each layer given input x."""
        # This method should be implemented based on your specific model architecture
        raise NotImplementedError("Implement based on your model architecture")


    @abstractmethod
    def _get_pre_activations(self, x: np.ndarray) -> List[np.ndarray]:
        """Calculate pre-activation values for each layer given input x."""
        raise NotImplementedError("Implement based on your model architecture")


    @abstractmethod
    def _calc_model_shape(self) -> Dict[str,int]:
        """
        Get the shape of each layer in the model.
        Returns: List of integers representing the number of neurons in each layer
        """
        raise NotImplementedError("Implement based on your model architecture")
    
    
    def split_correct_incorrect(self,x,y,mask):
        """
        Splits the dataset into correct and incorrect subsets based on predefined masks.
        Args:
            x[np.ndarray]: Input data.
            y[np.ndarray]: Labels.
            mask[np.ndarray]: Boolean mask for correct/incorrect predictions.
        Returns:
            dict: A dictionary with the following keys:
            - "x_correct": Correct inputs.
            - "x_incorrect": Incorrect inputs.
            - "y_correct": Correct labels.
            - "y_incorrect": Incorrect labels.
        """
        
        
        split =  {
            # x_train
            "x_correct": x[mask],
            "x_incorrect": x[~mask],
            
            # y_train
            "y_correct": y[mask],
            "y_incorrect": y[~mask],
        }
        
        return split


    def calc_selected_layers_list(self, selected_layers_str:str):
        """prepare selected layers (all vs hidden vs output etc) -legacy version is layers_used"""
        
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
if __name__ == '__main__':
    #Quick test to confirm it builds
    model_handler = ModelHandler()

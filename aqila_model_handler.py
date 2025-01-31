from typing import Tuple, Union, Dict, Any
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Optional


class ModelHandler(ABC):
    def __init__(self, dataset: Dict[str, np.ndarray], model: Any, ):
        super(ModelHandler, self).__init__()
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
            dataset: Dictionary containing training and testing data
            model: Pre-trained model instance
            selected_layers: List of layer indices to analyze
        """
        self.x_train = dataset["x_train"]
        self.y_train = dataset["y_train"]
        self.x_test = dataset["x_test"]
        self.y_test = dataset["y_test"]
        self.model = model

        # Calculate predictions
        self.y_train_pred = self._get_predictions(self.x_train)
        self.y_test_pred = self._get_predictions(self.x_test)
        
        # Calculate classification results
        self.train_correct_mask = self._get_correct_predictions_mask(self.y_train, self.y_train_pred)
        self.test_correct_mask = self._get_correct_predictions_mask(self.y_test, self.y_test_pred)

    @abstractmethod
    def _get_predictions(self, x: np.ndarray) -> np.ndarray:
        """Get model predictions for input x.
        Example:
        raw_predictions = self.model.predict(x)
        return np.argmax(raw_predictions, axis=1)
        """
        return None
    
    @abstractmethod
    def _get_correct_predictions_mask(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Create boolean mask of correct predictions."""
        if len(y_true.shape) > 1:  # Handle one-hot encoded labels
            y_true = np.argmax(y_true, axis=1)
        return y_true == y_pred
    
    @abstractmethod
    def _get_activations(self, x: np.ndarray) -> List[np.ndarray]:
        """Calculate activations for each layer given input x."""
        # This method should be implemented based on your specific model architecture
        # For Keras models, you might use something like:
        # return [layer_model.predict(x) for layer_model in self.layer_models]
        raise NotImplementedError("Implement based on your model architecture")
    
    @abstractmethod
    def _get_pre_activations(self, x: np.ndarray) -> List[np.ndarray]:
        """Calculate pre-activation values for each layer given input x."""
        # This method should be implemented based on your specific model architecture
        raise NotImplementedError("Implement based on your model architecture")

    @property
    def model_shape(self) -> List[int]:
        """Get the shape of each layer in the model."""
        # This method should be implemented based on your specific model architecture
        raise NotImplementedError("Implement based on your model architecture")
    
from abc import ABC, abstractmethod
import numpy as np
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from typing import Dict, List, Any


class ModelHandler(ABC):
    def __init__(self, dataset: Dict[str, List[str]], model_path: str, selected_layers: List[int] = None):
        """
        Abstract ModelHandler to analyze neural network layers.

        Args:
            dataset: Dictionary containing text data for training and testing.
            model_path: Path to fine-tuned .pth file.
            selected_layers: List of layer indices to analyze.
        """
        self.x_train = dataset["x_train"]
        self.y_train = np.array(dataset["y_train"])
        self.x_test = dataset["x_test"]
        self.y_test = np.array(dataset["y_test"])

        self.selected_layers = selected_layers if selected_layers else [0, 3, 5]  # Default layers

        # Load model
        self.tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
        self.model = self.load_model(model_path)

        # Compute predictions and accuracy masks
        self.y_train_pred = self._get_predictions(self.x_train)
        self.y_test_pred = self._get_predictions(self.x_test)
        self.train_correct_mask = self._get_correct_predictions_mask(self.y_train, self.y_train_pred)
        self.test_correct_mask = self._get_correct_predictions_mask(self.y_test, self.y_test_pred)

    @abstractmethod
    def load_model(self, model_path: str) -> Any:
        """Load the pre-trained model and return it."""
        pass

    @abstractmethod
    def _get_predictions(self, texts: List[str]) -> np.ndarray:
        """Generate predictions for input texts."""
        pass

    @abstractmethod
    def _get_correct_predictions_mask(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Create a boolean(i.e 0,1) mask for correct predictions."""
        pass

    @abstractmethod
    def _get_activations(self, text: str) -> List[np.ndarray]:
        """Compute activations for selected layers."""
        pass

    @abstractmethod
    def _get_pre_activations(self, text: str) -> List[np.ndarray]:
        """Compute pre-activation values for selected layers."""
        pass

    @property
    @abstractmethod
    def model_shape(self) -> List[int]:
        """Return the shape of each layer in the model."""
        pass

    def get_accuracy(self) -> float:
        """Compute accuracy based on input data."""
        correct_predictions = np.sum(self.test_correct_mask)
        total_predictions = len(self.y_test)
        return correct_predictions / total_predictions if total_predictions > 0 else 0.0

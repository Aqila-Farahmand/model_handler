import torch
import numpy as np
from typing import List, Dict, Any
from transformers import DistilBertForSequenceClassification, DistilBertModel
from abc import ABC, abstractmethod


class ModelHandler(ABC):
    def __init__(
            self,
            dataset: Dict[str, Dict[str, torch.Tensor]],  # Expect tokenized input
            model: DistilBertForSequenceClassification,
            device: str = "cpu"
    ):
        """
        Abstract Model Handler for DistilBERT Fake News Classification.

        Args:
            dataset: Dictionary containing tokenized training and testing data.
                     Should include 'input_ids' and 'attention_mask'.
            model: Pre-trained DistilBertForSequenceClassification instance.
            device: Device to run the model ("cpu" or "cuda").
        """
        self.device = device
        self.model = model.to(self.device)

        # Move dataset to device
        self.x_train = {k: v.to(self.device) for k, v in dataset["x_train"].items()}
        self.y_train = dataset["y_train"].to(self.device)
        self.x_test = {k: v.to(self.device) for k, v in dataset["x_test"].items()}
        self.y_test = dataset["y_test"].to(self.device)

        # Set model to evaluation mode
        self.model.eval()

    @abstractmethod
    def _get_predictions(self, x: Dict[str, torch.Tensor]) -> np.ndarray:
        """Get model predictions for input x."""
        pass

    @abstractmethod
    def _get_correct_predictions_mask(self, y_true: torch.Tensor, y_pred: np.ndarray) -> np.ndarray:
        """Create boolean mask of correct predictions."""
        pass

    @abstractmethod
    def _get_activations(self, x: Dict[str, torch.Tensor]) -> List[np.ndarray]:
        """Extract hidden states (activations) from DistilBERT layers."""
        pass

    @abstractmethod
    def _get_pre_activations(self, x: Dict[str, torch.Tensor]) -> List[np.ndarray]:
        """Extract logits before applying softmax (pre-activation values)."""
        pass

    @property
    @abstractmethod
    def model_shape(self) -> List[int]:
        """Return the number of hidden units per layer in DistilBERT."""
        pass


class FakeNewsModelHandler(ModelHandler):
    def _get_predictions(self, x: Dict[str, torch.Tensor]) -> np.ndarray:
        """Get model predictions using DistilBERT."""
        with torch.no_grad():
            logits = self.model(x["input_ids"], x["attention_mask"]).logits
            return torch.argmax(logits, dim=1).cpu().numpy()

    def _get_correct_predictions_mask(self, y_true: torch.Tensor, y_pred: np.ndarray) -> np.ndarray:
        """Create boolean mask of correct predictions."""
        y_true_np = y_true.cpu().numpy()
        return y_true_np == y_pred

    def _get_activations(self, x: Dict[str, torch.Tensor]) -> List[np.ndarray]:
        """Extract hidden states (activations) from DistilBERT model."""
        with torch.no_grad():
            hidden_states = self.model.distilbert(x["input_ids"], x["attention_mask"]).last_hidden_state
            return [layer.cpu().numpy() for layer in hidden_states]  # Convert to NumPy

    def _get_pre_activations(self, x: Dict[str, torch.Tensor]) -> List[np.ndarray]:
        """Extract logits before applying softmax (pre-activation values)."""
        with torch.no_grad():
            logits = self.model(x["input_ids"], x["attention_mask"]).logits
            return [logits.cpu().numpy()]  # Return as list to match activation format

    @property
    def model_shape(self) -> List[int]:
        """Return the number of hidden units per layer in DistilBERT."""
        return [layer.out_features for layer in self.model.distilbert.transformer.layer]


# usage:
if __name__ == "__main__":
    from transformers import DistilBertTokenizer

    # Load a pre-trained model and tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

    # tokenized dataset (mocked)
    dataset = {
        "x_train": tokenizer(["Fake news example", "True news example"], padding=True, return_tensors="pt"),
        "y_train": torch.tensor([0, 1]),  # Labels
        "x_test": tokenizer(["Another fake news", "Some real news"], padding=True, return_tensors="pt"),
        "y_test": torch.tensor([0, 1])
    }

    # Initialize and use the handler
    handler = FakeNewsModelHandler(dataset, model, device="cpu")
    preds = handler._get_predictions(handler.x_test)
    print("Predictions:", preds)

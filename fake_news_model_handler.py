import torch
import numpy as np
from typing import List, Dict, Any
from transformers import DistilBertForSequenceClassification
from abc import ABC


class FakeNewsModelHandler(ModelHandler, ABC):
    def __init__(
            self,
            dataset: Dict[str, Dict[str, torch.Tensor]],  # Expect tokenized input
            model: FakeNewsClassifier,
            device: str = "cpu"
    ):
        """
        Args:
            dataset: Dictionary containing tokenized training and testing data.
                     Should include 'input_ids' and 'attention_mask'.
            model: Pre-trained FakeNewsClassifier instance.
            device: Device to run the model ("cpu" or "cuda").
        """
        self.device = device
        self.model = model.to(self.device)

        # Move dataset to device
        self.x_train = {k: v.to(self.device) for k, v in dataset["x_train"].items()}
        self.y_train = dataset["y_train"].to(self.device)
        self.x_test = {k: v.to(self.device) for k, v in dataset["x_test"].items()}
        self.y_test = dataset["y_test"].to(self.device)

        # Set model to eval mode
        self.model.eval()

        # Compute predictions
        self.y_train_pred = self._get_predictions(self.x_train)
        self.y_test_pred = self._get_predictions(self.x_test)

        # Calculate classification results
        self.train_correct_mask = self._get_correct_predictions_mask(self.y_train, self.y_train_pred)
        self.test_correct_mask = self._get_correct_predictions_mask(self.y_test, self.y_test_pred)

    @torch.no_grad()
    def _get_predictions(self, x: Dict[str, torch.Tensor]) -> np.ndarray:
        """Get model predictions for input x."""
        logits = self.model(x["input_ids"], x["attention_mask"])
        return torch.argmax(logits, dim=1).cpu().numpy()

    def _get_correct_predictions_mask(self, y_true: torch.Tensor, y_pred: np.ndarray) -> np.ndarray:
        """Create boolean mask of correct predictions."""
        y_true_np = y_true.cpu().numpy()
        return y_true_np == y_pred

    @torch.no_grad()
    def _get_activations(self, x: Dict[str, torch.Tensor]) -> List[np.ndarray]:
        """Extract hidden states (activations) from DistilBERT layers."""
        outputs = self.model.bert(x["input_ids"], x["attention_mask"], output_hidden_states=True)
        hidden_states = outputs.hidden_states  # Tuple of (layer_count, batch_size, seq_len, hidden_dim)
        return [layer.cpu().numpy() for layer in hidden_states]

    @torch.no_grad()
    def _get_pre_activations(self, x: Dict[str, torch.Tensor]) -> List[np.ndarray]:
        """Extract logits before applying softmax (pre-activation values)."""
        logits = self.model(x["input_ids"], x["attention_mask"])
        return [logits.cpu().numpy()]

    @property
    def model_shape(self) -> List[int]:
        """Return the number of hidden units per layer in DistilBERT."""
        return [layer.shape[-1] for layer in self._get_activations(self.x_train)]
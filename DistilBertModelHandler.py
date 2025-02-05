from ModelHandler import ModelHandler
import numpy as np
import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
from typing import Dict, List, Any


class DistilBertModelHandler(ModelHandler):
    def __init__(self, dataset: Dict[str, List[str]], model_path: str, selected_layers: List[int] = None):
        """
        DistilBERT Model Handler for Sequence Classification.

        Args:
            dataset: Dictionary containing text data for training and testing.
            model_path: Path to fine-tuned (.pth file).
            selected_layers: List of layer indices to analyze.
        """
        super().__init__(dataset, model_path, selected_layers)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def load_model(self, model_path: str) -> DistilBertForSequenceClassification:
        """
        Load the fine-tuned DistilBERT model

        Args:
            model_path: Path to the fine-tuned model file (.pth).

        Returns:
            Loaded model.
        """
        # first, load the pre-trained DistilBERT model from HuggingFace
        model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")

        # then, load the fine-tuned weights from the .pth file
        model.load_state_dict(torch.load(model_path, map_location=self.device))

        # set the model in evaluation mode
        model.eval()

        return model

    def _get_predictions(self, texts: List[str]) -> np.ndarray:
        """
        Generate predictions for input texts using the model.

        Args:
            texts: List of input text sequences.

        Returns:
            Predictions as an array of integers (0 or 1).
        """
        # Tokenize input texts
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}


        # Predict
        with torch.no_grad():
            logits = self.model(**inputs).logits

        predictions = torch.argmax(logits, dim=-1).cpu().numpy()
        return predictions

    def _get_correct_predictions_mask(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Create a boolean mask for correct predictions.

        Args:
            y_true: Ground truth labels.
            y_pred: Predicted labels.

        Returns:
            Boolean mask indicating correct predictions.
            the return value is a boolean array with the same length as input array
        """
        return (y_true == y_pred)


    # To Do: get_activation() return type is dict [key: layer_name, value: list of ]
    def _get_activations(self, text: str) -> List[np.ndarray]:
        """
        Compute activations for selected layers of the model.

        Args:
            text: Input text for which activations will be computed.

        Returns:
            List of activations for selected layers.
        """
        # Tokenize input text
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=512)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        activations = []

        # Register hooks for selected layers
        # Hooks allows, to access the outputs (or inputs) of specific layers in the model
        def save_activation(layer_name):
            def hook(model, input, output):
                activations.append(output.cpu().detach().numpy())

            return hook

        # Register hooks for the selected layers
        hooks = []
        for layer_index in self.selected_layers:
            layer = self.model.distilbert.transformer.layer[layer_index]
            hook = layer.register_forward_hook(save_activation(layer))
            hooks.append(hook)

        # Perform a forward pass to get activations
        with torch.no_grad():
            self.model(**inputs)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        return activations

    # Get the values from the neural nodes BEFORE activation is done.

    def _get_pre_activations(self, text: str) -> List[np.ndarray]:
        """
        Compute pre-activation values for selected layers of the model.

        Args:
            text: Input text for which pre-activations will be computed.

        Returns:
            List of pre-activation values for selected layers.
        """
        # Tokenize input text
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=512)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        # Register hooks for selected layers, to get pre_activation values
        pre_activations = []

        def save_pre_activation(layer_name):
            def hook(model, input, output):
                # Pre-activation is the input to the layer before activation function
                pre_activations.append(input[0].cpu().detach().numpy())

            return hook

        # Register hooks for the selected layers
        hooks = []
        for layer_index in self.selected_layers:
            layer = self.model.distilbert.transformer.layer[layer_index]
            hook = layer.attention.q_lin.register_forward_hook(save_pre_activation("query"))
            hooks.append(hook)

        # Do a forward pass to get pre-activations
        with torch.no_grad():
            self.model(**inputs)

        # Remove hooks
        for hook in hooks:
            hook.remove()

        return pre_activations

    @property
    def model_shape(self) -> List[int]:
        """
        Return the shape of each layer in the model.

        Returns:
            List of integers representing the shape of each layer.
        """
        layer_shapes = [layer.attention.q_lin.weight.shape[0] for layer in self.model.distilbert.transformer.layer]
        return layer_shapes
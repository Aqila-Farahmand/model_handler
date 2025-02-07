from modelHandler import ModelHandler
import numpy as np
import torch
from typing import Dict, List, Any


class DistilBertModelHandler(ModelHandler):
    def __init__(self, selected_layers: str, use_pre_activation_values: bool, use_onehot_encoder: bool, model: Any ):
        """
        DistilBERT Model Handler for Sequence Classification.

        Args:
            model_path (str): Path to the fine-tuned model file (.pth).
            selected_layers (List[int], optional): List of layer indices to analyze.
        """
        super().__init__(selected_layers, use_pre_activation_values=True, use_onehot_encoder=False, model=None)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.model = model

    def _get_predictions(self, texts: List[str]) -> np.ndarray:
        """
        Generate predictions for input texts using the model.

        Args:
            texts (List[str]): List of input text sequences.

        Returns:
            np.ndarray: Predictions as an array of integers (0 or 1).
        """
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt", max_length=512)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits

        return torch.argmax(logits, dim=-1).cpu().numpy()

    def _get_correct_predictions_mask(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """
        Create a boolean mask for correct predictions.

        Args:
            y_true (np.ndarray): Ground truth labels.
            y_pred (np.ndarray): Predicted labels.

        Returns:
            np.ndarray: Boolean mask indicating correct predictions.
        """
        return y_true == y_pred

    def _get_activations(self, text: str) -> Dict[str, np.ndarray]:
        """
        Compute activations for selected layers of the model.

        Args:
            text (str): Input text for which activations will be computed.

        Returns:
            Dict[str, np.ndarray]: Dictionary with layer names as keys and activations as values.
        """
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=512)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        activations = {}

        def save_activation(layer_name):
            def hook(model, input, output):
                activations[layer_name] = output.cpu().detach().numpy()
            return hook

        hooks = []
        for layer_index in self.selected_layers:
            layer = self.model.distilbert.transformer.layer[layer_index]
            hook = layer.register_forward_hook(save_activation(f"layer_{layer_index}"))
            hooks.append(hook)

        with torch.no_grad():
            self.model(**inputs)

        for hook in hooks:
            hook.remove()

        return activations

    def _get_pre_activations(self, text: str) -> Dict[str, np.ndarray]:
        """
        Compute pre-activation values for selected layers.

        Args:
            text (str): Input text for which pre-activations will be computed.

        Returns:
            Dict[str, np.ndarray]: Dictionary with layer names as keys and pre-activations as values.
        """
        inputs = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt", max_length=512)
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        pre_activations = {}

        def save_pre_activation(layer_name):
            def hook(model, input, output):
                pre_activations[layer_name] = input[0].cpu().detach().numpy()
            return hook

        hooks = []
        for layer_index in self.selected_layers:
            layer = self.model.distilbert.transformer.layer[layer_index]
            hook = layer.attention.q_lin.register_forward_hook(save_pre_activation(f"pre_layer_{layer_index}"))
            hooks.append(hook)

        with torch.no_grad():
            self.model(**inputs)

        for hook in hooks:
            hook.remove()

        return pre_activations

    @property
    def model_shape(self) -> Dict[str, int]:
        """
        Return the shape of each layer in the model.

        Returns:
            Dict[str, int]: Dictionary with layer names as keys and neuron counts as values.
        """
        return {
            f"layer_{i}": layer.attention.q_lin.weight.shape[0] for i, layer in enumerate(self.model.distilbert.transformer.layer)
        }

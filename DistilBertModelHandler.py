import torch
from typing import Dict, List, Any
from ModelHandler import ModelHandler
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer


class DistilBertModelHandler(ModelHandler):
    def load_model(self, model_path: str) -> Any:
        """Load the fine-tuned DistilBERT model."""
        model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2
        )
        model.load_state_dict(torch.load(model_path, map_location="cpu"))
        model.eval()
        return model

    def _get_predictions(self, texts: List[str]) -> np.ndarray:
        """Generate predictions for input texts."""
        predictions = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
                pred = torch.argmax(outputs.logits, dim=1).item()
            predictions.append(pred)
        return np.array(predictions)

    def _get_correct_predictions_mask(self, y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
        """Create a boolean mask for correct predictions."""
        return y_true == y_pred

    def _get_activations(self, text: str) -> List[np.ndarray]:
        """Compute activations for selected layers."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
        activations = []
        hooks = []

        def hook_fn(module, input, output):
            activations.append(output.detach().numpy())

        for layer_idx in self.selected_layers:
            hook = self.model.distilbert.transformer.layer[layer_idx].register_forward_hook(hook_fn)
            hooks.append(hook)

        with torch.no_grad():
            _ = self.model(**inputs)

        for hook in hooks:
            hook.remove()

        return activations

    def _get_pre_activations(self, text: str) -> List[np.ndarray]:
        """Compute pre-activation values for selected layers."""
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding="max_length", max_length=512)
        pre_activations = []
        hooks = []

        def hook_fn(module, input, output):
            pre_activations.append(input[0].detach().numpy())

        for layer_idx in self.selected_layers:
            hook = self.model.distilbert.transformer.layer[layer_idx].register_forward_hook(hook_fn)
            hooks.append(hook)

        with torch.no_grad():
            _ = self.model(**inputs)

        for hook in hooks:
            hook.remove()

        return pre_activations

    @property
    def model_shape(self) -> List[int]:
        """Return the shape of each layer in the model."""
        return [768] * 6  # DistilBERT has 6 transformer layers, each with 768 neurons

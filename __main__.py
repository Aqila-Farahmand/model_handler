from DistilBertModelHandler import DistilBertModelHandler


# Example dataset
test_dataset = {
    "x_train": ["Some fake news", "Some real news"],
    "y_train": [0, 1],
    "x_test": ["Fake news example", "Real news example"],
    "y_test": [0, 1],
}

# Load model and analyze selected layers
model_handler = DistilBertModelHandler(test_dataset, "model/distilbert-fake-news.pth", selected_layers=[0, 2, 4])

# Print accuracy
print(f"Test Accuracy: {model_handler.get_accuracy() * 100:.2f}%")

# Get activations for a sample text
sample_text = "Breaking news: Something important happened!"
activations = model_handler._get_activations(sample_text)
pre_activations = model_handler._get_pre_activations(sample_text)

print(f"Number of selected layers: {len(activations)}")
print(f"Activation shape for layer 0: {activations[0].shape}")
print(f"Pre-activation shape for layer 0: {pre_activations[0].shape}")

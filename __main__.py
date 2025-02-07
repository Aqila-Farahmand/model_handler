from DistilBertModelHandler import DistilBertModelHandler
import torch
from transformers import DistilBertForSequenceClassification

test_dataset = {
    "x_train": [
        "Scientists discovered a new exoplanet orbiting a distant star.",
        "The government is introducing a new policy to reduce carbon emissions.",
        "Experts predict the global economy will grow by 3% this year.",
        "Aliens have been discovered living on Mars and are in contact with Earth.",
        "A recent report indicates a decline in the use of fossil fuels worldwide.",
        "A new species of bird has been discovered in the Amazon rainforest.",
        "A new cure for cancer is being suppressed by pharmaceutical companies.",
        "NASA is planning a mission to Mars in the next decade.",
        "A mysterious force is causing the oceans to rise and flood major cities worldwide.",
        "The president announced new measures to combat climate change."
    ],
    "y_train": [1, 1, 1, 0, 1, 1, 0, 1, 0, 1],  # 1 for true news, 0 for fake news
    "x_test": [
        "The world will end in 2025 due to a massive asteroid collision with Earth.",
        "A new medical treatment for Alzheimer's disease has been developed.",
        "A global government conspiracy to hide the truth about UFOs has been uncovered.",
        "Experts have confirmed the cure for all cancers has been found.",
        "A breakthrough in quantum computing was achieved by researchers."
    ],
    "y_test": [0, 1, 0, 0, 1]
}


# Load a fine-tuned model
model_path = "path/to/your/fine_tuned_model.pth"
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased")
model.load_state_dict(torch.load(model_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
model.eval()

# Initialize the handler
selected_layers = [0, 3, 5]
model_handler = DistilBertModelHandler(selected_layers, False, True, model)

# Sample texts
texts = ["This is a great product!", "I am disappointed with the service."]
true_labels = np.array([1, 0])  # Assume 1 = positive, 0 = negative

# Predictions
predictions = model_handler._get_predictions(texts)
print("Predictions:", predictions)

# Accuracy Mask
correct_mask = model_handler._get_correct_predictions_mask(true_labels, predictions)
print("Correct Predictions:", correct_mask)

# Layer Activations
activations = model_handler._get_activations("I love this movie!")
print("Activation Layer Names:", activations.keys())

# Pre-Activations
pre_activations = model_handler._get_pre_activations("I love this movie!")
print("Pre-Activation Layer Names:", pre_activations.keys())

# Model Layer Shapes
print("Model Layer Shapes:", model_handler.model_shape)


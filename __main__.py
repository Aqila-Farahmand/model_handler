from DistilBertModelHandler import DistilBertModelHandler

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


model_path = "model/distilbert-fake-news.pth"

# Load the model handler to analyze selected layers (0, 3, and 5)
model_handler = DistilBertModelHandler(test_dataset, model_path, selected_layers=[0, 3, 5])

test_accuracy = model_handler.get_accuracy()
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# sample text for calculating the activations and pre-activations
sample_text = "Breaking news: Something important happened!"

# calculate activations and pre-activations
try:
    activations = model_handler._get_activations(sample_text)
    pre_activations = model_handler._get_pre_activations(sample_text)

    print(f"Number of selected layers: {len(activations)}")
    for i, (act, pre_act) in enumerate(zip(activations, pre_activations)):
        print(f"\nLayer {i}:")
        print(f"  - Activation shape: {act.shape}")
        print(f"  - Pre-activation shape: {pre_act.shape}")
except Exception as e:
    print(f"Error while extracting activations and pre-activations: {e}")

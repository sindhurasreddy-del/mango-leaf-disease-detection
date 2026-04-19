"""
Mango Leaf Disease Prediction with Cure Recommendations
Classifies mango leaf images into 8 disease categories and provides treatment steps.
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

IMAGE_SIZE = 224

CLASS_NAMES = [
    'Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back',
    'Gall Midge', 'Healthy', 'Powdery Mildew', 'Sooty Mould'
]

CURE_STEPS = {
    'Anthracnose': [
        "Remove affected parts: Prune and destroy affected leaves and plant parts.",
        "Apply fungicides: Use fungicides containing chlorothalonil, copper, or mancozeb.",
        "Maintain sanitation: Clean the area around plants of plant debris.",
        "Water properly: Avoid overhead watering."
    ],
    'Bacterial Canker': [
        "Remove infected plants: Uproot and destroy infected plants.",
        "Use copper sprays: Apply copper-based bactericides.",
        "Sanitize tools: Disinfect pruning tools regularly.",
        "Avoid overhead irrigation: Water at the base of plants."
    ],
    'Cutting Weevil': [
        "Manual removal: Handpick weevils and larvae.",
        "Apply insecticides: Use insecticides like pyrethroids or neem oil.",
        "Use beneficial nematodes: Introduce beneficial nematodes in the soil."
    ],
    'Die Back': [
        "Pruning: Prune and destroy affected plant parts.",
        "Apply fungicides: Use appropriate fungicides.",
        "Improve air circulation: Ensure proper spacing of plants.",
        "Water management: Avoid overwatering and ensure good drainage."
    ],
    'Gall Midge': [
        "Remove galls: Prune and destroy affected plant parts.",
        "Apply insecticides: Use insecticides that target gall midges.",
        "Use sticky traps: Monitor and reduce midge populations with sticky traps."
    ],
    'Healthy': [
        "Regular monitoring: Inspect plants frequently.",
        "Balanced nutrition: Provide balanced fertilization.",
        "Water management: Water plants appropriately.",
        "Mulching: Use mulch to conserve moisture and reduce weed growth."
    ],
    'Powdery Mildew': [
        "Remove infected parts: Trim and dispose of infected leaves.",
        "Apply fungicides: Use fungicides like sulfur, neem oil, or potassium bicarbonate.",
        "Increase airflow: Space plants properly and prune to improve air circulation.",
        "Water at ground level: Avoid wetting foliage when watering."
    ],
    'Sooty Mould': [
        "Control insects: Manage sap-sucking insects like aphids and whiteflies.",
        "Clean leaves: Wash affected leaves with water.",
        "Use insecticidal soap: Apply insecticidal soaps or oils.",
        "Prune affected areas: Trim and remove heavily infested plant parts."
    ]
}


def predict(model, img):
    """Predict disease class and confidence from a leaf image."""
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.image.resize(img_array, (IMAGE_SIZE, IMAGE_SIZE))
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = round(100 * np.max(predictions[0]), 2)
    return predicted_class, confidence


def load_and_predict_with_cure_steps(model_path, image_path):
    """Load model, predict disease, display image with cure recommendations."""
    model = tf.keras.models.load_model(model_path)

    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = tf.keras.preprocessing.image.img_to_array(img)

    predicted_class, confidence = predict(model, img_array)
    cure = CURE_STEPS.get(predicted_class, ["No specific cure steps found."])

    plt.imshow(img)
    plt.title(f"Predicted: {predicted_class}, Confidence: {confidence}%")
    plt.axis("off")
    plt.show()

    print(f"\nDisease: {predicted_class}")
    print(f"Confidence: {confidence}%")
    print("\nCurative steps:")
    for i, step in enumerate(cure, 1):
        print(f"  {i}. {step}")

    return predicted_class, confidence, cure


if __name__ == '__main__':
    import sys
    if len(sys.argv) != 3:
        print("Usage: python predict.py <model_path> <image_path>")
        print("Example: python predict.py saved/model.h5 path/to/leaf.jpg")
        sys.exit(1)

    load_and_predict_with_cure_steps(sys.argv[1], sys.argv[2])

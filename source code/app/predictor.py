import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_DIR = BASE_DIR / "models"


# =========================================================
# LOAD MODELS
# =========================================================

component_model = tf.keras.models.load_model(
    MODEL_DIR / "component_classifier.keras"
)

winding_model = tf.keras.models.load_model(
    MODEL_DIR / "winding_fault_classifier.keras"
)


# =========================================================
# LOAD CLASS NAMES
# =========================================================

with open(
    MODEL_DIR / "component_class_names.json",
    "r"
) as f:

    component_classes = json.load(f)


with open(
    MODEL_DIR / "winding_class_names.json",
    "r"
) as f:

    winding_classes = json.load(f)


# Add to app/predictor.py

HEALTH_RULES = {
    # Healthy reference condition
    "Noload": {
        "score": 100,
        "reason": "Healthy reference condition in the dataset."
    },

    # Stator winding short-circuit conditions
    "A10": {
        "score": 90,
        "reason": "Lower labelled winding-fault severity."
    },
    "A&C10": {
        "score": 85,
        "reason": "Lower labelled severity involving two phases."
    },
    "A&C&B10": {
        "score": 80,
        "reason": "Lower labelled severity involving three phases."
    },
    "A30": {
        "score": 70,
        "reason": "Moderate labelled winding-fault severity."
    },
    "A&C30": {
        "score": 65,
        "reason": "Moderate labelled severity involving two phases."
    },
    "A&C&B30": {
        "score": 60,
        "reason": "Moderate labelled severity involving three phases."
    },
    "A50": {
        "score": 50,
        "reason": "Higher labelled winding-fault severity."
    },
    "A&B50": {
        "score": 45,
        "reason": "Higher labelled severity involving two phases."
    },

    # These are fault categories, not calibrated severity levels.
    "Fan": {
        "score": 40,
        "reason": "Cooling-fan fault category detected."
    },
    "Rotor-0": {
        "score": 30,
        "reason": "Rotor fault category detected."
    }
}


def estimate_health(predicted_class):
    """
    Prototype rule-based score.
    This is NOT a measured or validated percentage of motor health.
    """

    rule = HEALTH_RULES.get(predicted_class)

    if rule is None:
        return {
            "score": None,
            "status": "Unrated",
            "reason": "No health rule is configured for this condition.",
            "method": "rule_based_prototype"
        }

    score = rule["score"]

    if score >= 90:
        status = "Healthy reference"
    elif score >= 70:
        status = "Lower estimated concern"
    elif score >= 50:
        status = "Elevated estimated concern"
    else:
        status = "High estimated concern"

    return {
        "score": score,
        "status": status,
        "reason": rule["reason"],
        "method": "rule_based_prototype"
    }

# =========================================================
# IMAGE PREPROCESSING
# =========================================================

IMG_SIZE = (224, 224)


def preprocess_image(image):

    image = image.convert("RGB")

    image = image.resize(
        IMG_SIZE
    )

    image = np.array(
        image
    ).astype(
        np.float32
    )

    image = np.expand_dims(
        image,
        axis=0
    )

    return image


# =========================================================
# HIERARCHICAL PREDICTION
# =========================================================

def predict_image(image):

    processed = preprocess_image(
        image
    )


    # -----------------------------------------------------
    # STAGE 1
    # -----------------------------------------------------

    component_predictions = (
        component_model.predict(
            processed,
            verbose=0
        )[0]
    )


    component_index = int(
        np.argmax(
            component_predictions
        )
    )


    component = component_classes[
        component_index
    ]


    component_confidence = float(
        component_predictions[
            component_index
        ]
    )


    # -----------------------------------------------------
    # STAGE 2
    # -----------------------------------------------------

    if component == "Winding":

        winding_predictions = (
            winding_model.predict(
                processed,
                verbose=0
            )[0]
        )


        fault_index = int(
            np.argmax(
                winding_predictions
            )
        )


        fault_class = winding_classes[
            fault_index
        ]


        fault_confidence = float(
            winding_predictions[
                fault_index
            ]
        )


        probabilities = {

            cls: float(prob)

            for cls, prob in zip(
                winding_classes,
                winding_predictions
            )

        }


        return {

            "component": component,

            "component_confidence":
                component_confidence,

            "predicted_class":
                fault_class,

            "confidence":
                fault_confidence,

            "probabilities":
                probabilities,

            "stage":
                "winding_fault_classifier"

        }


    # -----------------------------------------------------
    # FAN
    # -----------------------------------------------------

    if component == "Fan":

        return {

            "component": "Fan",

            "component_confidence":
                component_confidence,

            "predicted_class":
                "Fan",

            "confidence":
                component_confidence,

            "probabilities": {
                "Fan":
                    component_confidence
            },

            "stage":
                "component_classifier"

        }


    # -----------------------------------------------------
    # ROTOR
    # -----------------------------------------------------

    if component == "Rotor":

        return {

            "component": "Rotor",

            "component_confidence":
                component_confidence,

            "predicted_class":
                "Rotor-0",

            "confidence":
                component_confidence,

            "probabilities": {
                "Rotor-0":
                    component_confidence
            },

            "stage":
                "component_classifier"

        }


    # -----------------------------------------------------
    # NORMAL
    # -----------------------------------------------------

    return {

        "component": "Normal",

        "component_confidence":
            component_confidence,

        "predicted_class":
            "Noload",

        "confidence":
            component_confidence,

        "probabilities": {
            "Noload":
                component_confidence
        },

        "stage":
            "component_classifier"

    }
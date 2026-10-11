import tensorflow as tf
import numpy as np


def make_gradcam_heatmap(
    image,
    model,
    last_conv_layer_name="top_conv",
    pred_index=None
):

    # =====================================================
    # GET EFFICIENTNET BACKBONE
    # =====================================================

    base_model = model.get_layer(
        "efficientnetb0"
    )


    last_conv_layer = (
        base_model.get_layer(
            last_conv_layer_name
        )
    )


    # =====================================================
    # GRADIENT MODEL
    # =====================================================

    grad_model = tf.keras.Model(

        inputs=base_model.input,

        outputs=[
            last_conv_layer.output,
            base_model.output
        ]

    )


    # =====================================================
    # FORWARD PASS
    # =====================================================

    with tf.GradientTape() as tape:

        conv_outputs, base_output = (
            grad_model(
                image,
                training=False
            )
        )


        x = model.layers[-3](
            base_output
        )


        x = model.layers[-2](
            x
        )


        predictions = model.layers[-1](
            x
        )


        if pred_index is None:

            pred_index = tf.argmax(
                predictions[0]
            )


        class_channel = predictions[
            :,
            pred_index
        ]


    # =====================================================
    # GRADIENTS
    # =====================================================

    grads = tape.gradient(
        class_channel,
        conv_outputs
    )


    pooled_grads = tf.reduce_mean(
        grads,
        axis=(0, 1, 2)
    )


    conv_outputs = conv_outputs[0]


    heatmap = (
        conv_outputs
        @ pooled_grads[..., tf.newaxis]
    )


    heatmap = tf.squeeze(
        heatmap
    )


    heatmap = tf.maximum(
        heatmap,
        0
    )


    max_value = tf.reduce_max(
        heatmap
    )


    heatmap = tf.where(
        max_value > 0,
        heatmap / max_value,
        heatmap
    )


    return (
        heatmap.numpy(),
        int(pred_index)
    )

import cv2


def create_gradcam_overlay(
    image,
    model,
    predicted_index
):

    # -----------------------------------------------------
    # Resize input
    # -----------------------------------------------------

    original = image.convert(
        "RGB"
    )

    original = original.resize(
        (224, 224)
    )


    original_np = np.array(
        original
    )


    input_tensor = np.expand_dims(
        original_np.astype(
            np.float32
        ),
        axis=0
    )


    # -----------------------------------------------------
    # Heatmap
    # -----------------------------------------------------

    heatmap, _ = make_gradcam_heatmap(

        input_tensor,

        model,

        last_conv_layer_name="top_conv",

        pred_index=predicted_index

    )


    # -----------------------------------------------------
    # Resize heatmap
    # -----------------------------------------------------

    heatmap = cv2.resize(
        heatmap,
        (224, 224)
    )


    heatmap = np.uint8(
        255 * heatmap
    )


    # -----------------------------------------------------
    # Apply color map
    # -----------------------------------------------------

    heatmap_color = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )


    # OpenCV BGR → RGB

    heatmap_color = cv2.cvtColor(
        heatmap_color,
        cv2.COLOR_BGR2RGB
    )


    # -----------------------------------------------------
    # Overlay
    # -----------------------------------------------------

    overlay = cv2.addWeighted(

        original_np,

        0.55,

        heatmap_color,

        0.45,

        0

    )


    return overlay
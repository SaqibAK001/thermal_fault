import base64
import io

from pathlib import Path

from fastapi import (
    FastAPI,
    File,
    UploadFile,
    HTTPException
)

from fastapi.responses import (
    HTMLResponse,
    JSONResponse
)

from fastapi.staticfiles import StaticFiles

from fastapi.templating import Jinja2Templates

from fastapi import Request

from PIL import Image

from app.predictor import (
    predict_image,
    estimate_health,
    component_model,
    winding_model
)

from app.gradcam import (
    create_gradcam_overlay
)


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(
    __file__
).resolve().parent.parent


TEMPLATES_DIR = (
    BASE_DIR / "templates"
)

STATIC_DIR = (
    BASE_DIR / "static"
)


# =========================================================
# APP
# =========================================================

app = FastAPI(
    title="ThermalGuard AI",
    description="Hierarchical Thermal Fault Detection System",
    version="2.0.0"
)


# =========================================================
# STATIC FILES
# =========================================================

app.mount(
    "/static",
    StaticFiles(
        directory=str(STATIC_DIR)
    ),
    name="static"
)


# =========================================================
# TEMPLATES
# =========================================================

templates = Jinja2Templates(
    directory=str(
        TEMPLATES_DIR
    )
)


# =========================================================
# HOME
# =========================================================

@app.get(
    "/",
    response_class=HTMLResponse
)
async def home(
    request: Request
):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )


# =========================================================
# HEALTH
# =========================================================

@app.get("/api/health")
async def health():

    return {

        "status": "online",

        "system":
            "ThermalGuard AI",

        "architecture":
            "Hierarchical",

        "stage_1":
            "Component Classifier",

        "stage_2":
            "Winding Fault Classifier",

        "explainability":
            "Grad-CAM"

    }


# =========================================================
# PREDICT
# =========================================================

@app.post(
    "/api/predict"
)
async def predict(
    file: UploadFile = File(...)
):

    # -----------------------------------------------------
    # Validate file
    # -----------------------------------------------------

    if not file.content_type:

        raise HTTPException(
            status_code=400,
            detail="Invalid file."
        )


    if not file.content_type.startswith(
        "image/"
    ):

        raise HTTPException(
            status_code=400,
            detail="Please upload an image."
        )


    # -----------------------------------------------------
    # Read image
    # -----------------------------------------------------

    contents = await file.read()


    try:

        image = Image.open(
            io.BytesIO(contents)
        ).convert("RGB")

    except Exception:

        raise HTTPException(
            status_code=400,
            detail="Could not read image."
        )


    # -----------------------------------------------------
    # Hierarchical prediction
    # -----------------------------------------------------

    result = predict_image(image)

    health = estimate_health(result["predicted_class"])


    # -----------------------------------------------------
    # Determine model used
    # -----------------------------------------------------

    if result["stage"] == (
        "winding_fault_classifier"
    ):

        used_model = winding_model

    else:

        used_model = component_model


    # -----------------------------------------------------
    # Grad-CAM
    # -----------------------------------------------------

    try:

        predicted_index = 0


        if result["stage"] == (
            "winding_fault_classifier"
        ):

            from app.predictor import (
                winding_classes
            )

            predicted_index = (
                winding_classes.index(
                    result["predicted_class"]
                )
            )

        else:

            from app.predictor import (
                component_classes
            )

            predicted_index = (
                component_classes.index(
                    result["component"]
                )
            )


        overlay = (
            create_gradcam_overlay(

                image,

                used_model,

                predicted_index

            )
        )


        # -------------------------------------------------
        # Convert overlay to JPEG
        # -------------------------------------------------

        overlay_image = Image.fromarray(
            overlay
        )


        buffer = io.BytesIO()


        overlay_image.save(
            buffer,
            format="JPEG",
            quality=90
        )


        gradcam_base64 = base64.b64encode(
            buffer.getvalue()
        ).decode("utf-8")


    except Exception as e:

        print(
            "Grad-CAM error:",
            e
        )

        gradcam_base64 = None


    # -----------------------------------------------------
    # Response
    # -----------------------------------------------------

    response = {

        "component":
            result["component"],

        "component_confidence":
            round(
                result[
                    "component_confidence"
                ] * 100,
                2
            ),

        "predicted_class":
            result[
                "predicted_class"
            ],

        "confidence_percent":
            round(
                result["confidence"] * 100,
                2
            ),

        "probabilities":
            result["probabilities"],

        "stage":
            result["stage"],

        "gradcam":
            gradcam_base64,

        "health_score": health["score"],
        "health_status": health["status"],
        "health_reason": health["reason"],
        "health_method": health["method"]

    }


    return JSONResponse(
        content=response
    )
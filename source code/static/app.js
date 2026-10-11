// =========================================================
// ELEMENTS
// =========================================================

const imageInput = document.getElementById("imageInput");

const browseBtn = document.getElementById("browseBtn");

const dropZone = document.getElementById("dropZone");

const fileName = document.getElementById("fileName");

const previewContainer =
    document.getElementById("previewContainer");

const imagePreview =
    document.getElementById("imagePreview");

const removeImage =
    document.getElementById("removeImage");

const analyzeBtn =
    document.getElementById("analyzeBtn");

const analyzeText =
    document.getElementById("analyzeText");

const loader =
    document.getElementById("loader");

const resultsSection =
    document.getElementById("resultsSection");

const predictedClass =
    document.getElementById("predictedClass");

const confidenceValue =
    document.getElementById("confidenceValue");

const confidenceFill =
    document.getElementById("confidenceFill");

const conditionStatus =
    document.getElementById("conditionStatus");

const originalImage =
    document.getElementById("originalImage");

const gradcamImage =
    document.getElementById("gradcamImage");

const probabilities =
    document.getElementById("probabilities");


// =========================================================
// SELECT IMAGE
// =========================================================

browseBtn.addEventListener(
    "click",
    () => {
        imageInput.click();
    }
);


imageInput.addEventListener(
    "change",
    () => {

        if (imageInput.files.length > 0) {

            handleFile(
                imageInput.files[0]
            );
        }
    }
);


// =========================================================
// DRAG & DROP
// =========================================================

dropZone.addEventListener(
    "dragover",
    (event) => {

        event.preventDefault();

        dropZone.classList.add(
            "dragover"
        );
    }
);


dropZone.addEventListener(
    "dragleave",
    () => {

        dropZone.classList.remove(
            "dragover"
        );
    }
);


dropZone.addEventListener(
    "drop",
    (event) => {

        event.preventDefault();

        dropZone.classList.remove(
            "dragover"
        );

        const files =
            event.dataTransfer.files;

        if (files.length > 0) {

            const file = files[0];

            if (
                file.type.startsWith(
                    "image/"
                )
            ) {

                imageInput.files =
                    event.dataTransfer.files;

                handleFile(file);

            } else {

                alert(
                    "Please select an image file."
                );
            }
        }
    }
);


// =========================================================
// HANDLE FILE
// =========================================================

function handleFile(file) {

    fileName.textContent =
        file.name;

    const reader =
        new FileReader();

    reader.onload =
        function (event) {

            imagePreview.src =
                event.target.result;

            previewContainer.hidden =
                false;

            analyzeBtn.disabled =
                false;
        };

    reader.readAsDataURL(file);
}


// =========================================================
// REMOVE IMAGE
// =========================================================

removeImage.addEventListener(
    "click",
    () => {

        imageInput.value = "";

        fileName.textContent =
            "No image selected";

        imagePreview.src = "";

        previewContainer.hidden =
            true;

        analyzeBtn.disabled =
            true;

        resultsSection.hidden =
            true;
    }
);


// =========================================================
// ANALYZE IMAGE
// =========================================================

analyzeBtn.addEventListener(
    "click",
    analyzeImage
);


async function analyzeImage() {

    if (
        !imageInput.files.length
    ) {

        return;
    }


    // -----------------------------------------------------
    // Loading state
    // -----------------------------------------------------

    analyzeBtn.disabled =
        true;

    analyzeText.textContent =
        "Analyzing Thermal Image...";

    loader.hidden =
        false;


    // -----------------------------------------------------
    // Prepare form data
    // -----------------------------------------------------

    const formData =
        new FormData();

    formData.append(
        "file",
        imageInput.files[0]
    );


    try {

        // -------------------------------------------------
        // API request
        // -------------------------------------------------

        const response =
            await fetch(
                "/api/predict",
                {
                    method: "POST",
                    body: formData
                }
            );


        if (!response.ok) {

            throw new Error(
                "Prediction request failed."
            );
        }


        const data =
            await response.json();


        // -------------------------------------------------
        // Display results
        // -------------------------------------------------

        displayResults(data);


    } catch (error) {

        console.error(error);

        alert(
            "An error occurred while analyzing the image."
        );

    } finally {

        analyzeBtn.disabled =
            false;

        analyzeText.textContent =
            "Analyze Thermal Image";

        loader.hidden =
            true;
    }
}



function displayHealth(data) {
    const scoreElement = document.getElementById("healthScore");
    const barElement = document.getElementById("healthBar");
    const statusElement = document.getElementById("healthStatus");
    const reasonElement = document.getElementById("healthReason");

    if (
        data.health_score === null ||
        data.health_score === undefined
    ) {
        scoreElement.textContent = "--";
        barElement.style.width = "0%";
        statusElement.textContent = "Not rated";
        reasonElement.textContent =
            data.health_reason || "No health estimate is available.";
        return;
    }

    const score = Math.max(
        0,
        Math.min(100, Number(data.health_score))
    );

    scoreElement.textContent = score.toFixed(0);
    barElement.style.width = `${score}%`;
    statusElement.textContent =
        data.health_status || "Estimated condition";
    reasonElement.textContent =
        data.health_reason || "Rule-based prototype estimate.";

    barElement.classList.remove(
        "health-high",
        "health-medium",
        "health-low"
    );

    if (score >= 80) {
        barElement.classList.add("health-high");
    } else if (score >= 50) {
        barElement.classList.add("health-medium");
    } else {
        barElement.classList.add("health-low");
    }
}


// =========================================================
// DISPLAY RESULTS
// =========================================================

function displayResults(data) {

    resultsSection.hidden =
        false;


    // -----------------------------------------------------
    // Prediction
    // -----------------------------------------------------

    const modelStage =
        document.getElementById(
            "modelStage"
        );

    if (modelStage) {

        if (
            data.stage ===
            "winding_fault_classifier"
        ) {

            modelStage.textContent =
                "Stage 2 • Winding Fault Classifier";

        } else {

            modelStage.textContent =
                "Stage 1 • Component Classifier";
        }
    }

    predictedClass.textContent =
        data.predicted_class;

    const componentInfo =
        document.getElementById(
            "componentInfo"
        );

    if (componentInfo) {

        componentInfo.textContent =
            "Component: " +
            data.component +
            " • " +
            data.component_confidence.toFixed(2) +
            "% confidence";
    }


    // -----------------------------------------------------
    // Confidence
    // -----------------------------------------------------

    const confidence =
        data.confidence_percent;


    confidenceValue.textContent =
        confidence.toFixed(2) + "%";


    setTimeout(
        () => {

            confidenceFill.style.width =
                confidence + "%";

        },
        100
    );


    // -----------------------------------------------------
    // Condition status
    // -----------------------------------------------------

    if (
        data.predicted_class.toLowerCase()
            .includes("noload")
    ) {

        conditionStatus.textContent =
            "Normal / No-load condition";

        conditionStatus.style.color =
            "#35e69a";

    } else {

        conditionStatus.textContent =
            "Thermal condition detected";

        conditionStatus.style.color =
            "#ffc857";
    }


    // -----------------------------------------------------
    // Original image
    // -----------------------------------------------------

    originalImage.src =
        imagePreview.src;


    // -----------------------------------------------------
    // Grad-CAM
    // -----------------------------------------------------
    //
    // IMPORTANT:
    // FastAPI returns raw Base64.
    // We add the JPEG data URL prefix here.
    // -----------------------------------------------------

    if (data.gradcam) {

        gradcamImage.src =
            "data:image/jpeg;base64," +
            data.gradcam;

    } else {

        gradcamImage.alt =
            "Grad-CAM unavailable";
    }


    // -----------------------------------------------------
    // Class probabilities
    // -----------------------------------------------------

    probabilities.innerHTML =
        "";


    const sortedProbabilities =
        Object.entries(
            data.probabilities
        ).sort(
            (a, b) => b[1] - a[1]
        );


    sortedProbabilities.forEach(
        ([className, probability]) => {

            const percentage =
                probability * 100;


            const row =
                document.createElement(
                    "div"
                );

            row.className =
                "probability-row";


            row.innerHTML = `

                <div class="probability-header">

                    <span class="probability-name">
                        ${className}
                    </span>

                    <span class="probability-value">
                        ${percentage.toFixed(2)}%
                    </span>

                </div>

                <div class="probability-track">

                    <div
                        class="probability-fill"
                        style="width: ${percentage}%"
                    ></div>

                </div>

            `;


            probabilities.appendChild(
                row
            );
        }
    );


    // -----------------------------------------------------
    // Scroll to results
    // -----------------------------------------------------

    setTimeout(
        () => {

            resultsSection.scrollIntoView({
                behavior: "smooth",
                block: "start"
            });

        },
        100
    );
}
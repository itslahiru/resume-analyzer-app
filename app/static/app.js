const form = document.getElementById("analyze-form");
const loadingOverlay = document.getElementById("loading-overlay");
const analyzeButton = document.getElementById("analyze-button");
const fileInput = document.getElementById("file-input");
const fileNameText = document.getElementById("file-name");

if (fileInput && fileNameText) {
    fileInput.addEventListener("change", () => {
        if (fileInput.files.length > 0) {
            fileNameText.textContent = `Selected file: ${fileInput.files[0].name}`;
        } else {
            fileNameText.textContent = "No file selected";
        }
    });
}

if (form && loadingOverlay && analyzeButton) {
    form.addEventListener("submit", () => {
        loadingOverlay.classList.remove("hidden");
        analyzeButton.disabled = true;
        analyzeButton.textContent = "Analyzing...";
    });
}
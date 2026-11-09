
import { useState } from "react";
import "./App.css";

// Local backend URL
//const API_URL = "http://127.0.0.1:8000/predict";
const API_URL = "https://imagetruth-2.onrender.com/predict";

function ScoreBars({ scores }) {
  return (
    <div className="scores-section">
      <div className="score-row">
        <span>AI / Fake score</span>
        <strong>{scores?.fake ?? "—"}%</strong>
      </div>

      <div className="score-track">
        <div
          className="score-fill fake-fill"
          style={{
            width: `${Math.min(100, Math.max(0, Number(scores?.fake) || 0))}%`,
          }}
        />
      </div>

      <div className="score-row">
        <span>Real score</span>
        <strong>{scores?.real ?? "—"}%</strong>
      </div>

      <div className="score-track">
        <div
          className="score-fill real-fill"
          style={{
            width: `${Math.min(100, Math.max(0, Number(scores?.real) || 0))}%`,
          }}
        />
      </div>
    </div>
  );
}

function GradCAMView({ modelName, data }) {
  if (!data?.gradcam_heatmap || !data?.gradcam_overlay) {
    return null;
  }

  return (
    <div className="gradcam-section">
      <div className="subsection-heading">
        <h3>{modelName} Grad-CAM</h3>
        <span>Visual explanation</span>
      </div>

      <p className="section-description">
        These highlighted regions show which image areas influenced
        this model's prediction. They are not proof of image origin.
      </p>

      <div className="gradcam-grid">
        <div className="gradcam-item">
          <img
            src={`data:image/png;base64,${data.gradcam_heatmap}`}
            alt={`${modelName} Grad-CAM heatmap`}
          />
          <p>Heatmap</p>
        </div>

        <div className="gradcam-item">
          <img
            src={`data:image/png;base64,${data.gradcam_overlay}`}
            alt={`${modelName} Grad-CAM overlay`}
          />
          <p>Overlay</p>
        </div>
      </div>
    </div>
  );
}

function ModelCard({ title, data }) {
  if (!data) return null;

  const isFake = data.prediction?.toLowerCase() === "fake";

  return (
    <article className="feature-card model-result-card">
      <div className="subsection-heading">
        <h3>{title}</h3>
        <span>Individual model</span>
      </div>

      <div className={`prediction-banner ${isFake ? "fake" : "real"}`}>
        <div>
          <span className="result-overline">MODEL PREDICTION</span>
          <h3>
            {isFake ? "AI / Fake predicted" : "Real predicted"}
          </h3>
        </div>

        <span className="prediction-symbol">
          {isFake ? "!" : "✓"}
        </span>
      </div>

      <ScoreBars scores={data.scores} />

      <GradCAMView modelName={title} data={data} />
    </article>
  );
}

function App() {
  const [image, setImage] = useState(null);
  const [preview, setPreview] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  function handleImageChange(event) {
    const file = event.target.files?.[0];

    if (!file) return;

    if (!file.type.startsWith("image/")) {
      setError("Please select a valid image file.");
      return;
    }

    if (preview) {
      URL.revokeObjectURL(preview);
    }

    setImage(file);
    setPreview(URL.createObjectURL(file));
    setResult(null);
    setError("");
  }

  async function analyzeImage(event) {
    event.preventDefault();

    if (!image) {
      setError("Please upload an image first.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    const formData = new FormData();
    formData.append("image", image);

    try {
      const response = await fetch(API_URL, {
        method: "POST",
        body: formData,
      });

      const data = await response.json();

      if (!response.ok) {
        throw new Error(
          data.detail || "Image analysis failed. Please try again."
        );
      }

      // Verify the new backend response structure.
      if (
        !data.ensemble ||
        !data.efficientnet ||
        !data.resnet50
      ) {
        throw new Error(
          "The backend response is missing model comparison results. " +
          "Please check that the updated FastAPI backend is running."
        );
      }

      setResult(data);
    } catch (err) {
      setError(
        err.message ||
          "Cannot connect to the backend. Make sure FastAPI is running."
      );
    } finally {
      setLoading(false);
    }
  }

  function resetAnalysis() {
    if (preview) {
      URL.revokeObjectURL(preview);
    }

    setImage(null);
    setPreview("");
    setResult(null);
    setError("");

    const input = document.getElementById("image-upload");

    if (input) {
      input.value = "";
    }
  }

  const ensemblePrediction = result?.ensemble?.prediction;
  const isEnsembleFake =
    ensemblePrediction?.toLowerCase() === "fake";

  return (
    <div className="app">
      <header className="navbar">
        <a className="brand" href="#home">
          <span className="brand-icon">◎</span>
          ImageTruth
        </a>

        <span className="nav-label">AI Image Analysis</span>
      </header>

      <main id="home" className="main-container">
        {/* HERO SECTION */}

        <section className="hero">
          <span className="eyebrow">
            DEEP LEARNING · ENSEMBLE LEARNING · EXPLAINABLE AI
          </span>

          <h1>
            Discover the story
            <br />
            <span>behind an image.</span>
          </h1>

          <p className="hero-description">
            Analyze images using EfficientNet-B0 and ResNet50,
            compare their predictions, and explore the regions
            that influenced their decisions using Grad-CAM.
          </p>
        </section>

        {/* MAIN WORKSPACE */}

        <section className="workspace">
          {/* IMAGE UPLOAD */}

          <div className="upload-card">
            <div className="section-heading">
              <span className="step-number">01</span>

              <div>
                <h2>Upload an image</h2>
                <p>Choose a photo for analysis.</p>
              </div>
            </div>

            <form onSubmit={analyzeImage}>
              <label className="upload-area" htmlFor="image-upload">
                {preview ? (
                  <img
                    className="preview-image"
                    src={preview}
                    alt="Selected image preview"
                  />
                ) : (
                  <div className="upload-placeholder">
                    <span className="upload-icon">↑</span>
                    <strong>Click to select an image</strong>
                    <span>PNG, JPG, JPEG, or WEBP</span>
                  </div>
                )}
              </label>

              <input
                id="image-upload"
                type="file"
                accept="image/png,image/jpeg,image/webp"
                onChange={handleImageChange}
                hidden
              />

              {image && (
                <p className="file-name">
                  Selected: {image.name}
                </p>
              )}

              <button
                className="primary-button"
                type="submit"
                disabled={!image || loading}
              >
                {loading ? (
                  <>
                    <span className="spinner" />
                    Analyzing with both models...
                  </>
                ) : (
                  "Analyze Image →"
                )}
              </button>
            </form>

            {error && (
              <div className="error-message" role="alert">
                {error}
              </div>
            )}

            <p className="privacy-note">
              Your image is sent to your configured backend for analysis.
            </p>
          </div>

          {/* RESULTS */}

          <div className="results-card">
            <div className="section-heading">
              <span className="step-number">02</span>

              <div>
                <h2>Detection results</h2>
                <p>Compare both models and their ensemble prediction.</p>
              </div>
            </div>

            {!result && (
              <div className="empty-results">
                <div className="empty-icon">⌕</div>

                <h3>Results will appear here</h3>

                <p>
                  Upload an image and select Analyze Image to see
                  the individual model predictions, ensemble result,
                  and Grad-CAM explanations.
                </p>
              </div>
            )}

            {result && (
              <div className="result-content">

                {/* ENSEMBLE PREDICTION */}

                <section className="ensemble-section">
                  <div className="subsection-heading">
                    <h3>Ensemble prediction</h3>
                    <span>Combined result</span>
                  </div>

                  <div
                    className={`prediction-banner ${
                      isEnsembleFake ? "fake" : "real"
                    }`}
                  >
                    <div>
                      <span className="result-overline">
                        FINAL MODEL PREDICTION
                      </span>

                      <h3>
                        {isEnsembleFake
                          ? "AI / Fake predicted"
                          : "Real predicted"}
                      </h3>
                    </div>

                    <span className="prediction-symbol">
                      {isEnsembleFake ? "!" : "✓"}
                    </span>
                  </div>

                  <ScoreBars scores={result.ensemble.scores} />

                  <p className="section-description">
                    Method: {result.ensemble.method}
                  </p>
                </section>

                {/* MODEL AGREEMENT */}

                <div className="explanation-box">
                  <h3>
                    <span>✳</span> Model comparison
                  </h3>

                  <p>
                    {result.model_agreement
                      ? "EfficientNet-B0 and ResNet50 agree on the predicted class."
                      : "EfficientNet-B0 and ResNet50 disagree on the predicted class. The ensemble combines their scores using equal-weight averaging."}
                  </p>

                  <p>
                    <strong>EfficientNet-B0:</strong>{" "}
                    {result.efficientnet.prediction.toUpperCase()}
                  </p>

                  <p>
                    <strong>ResNet50:</strong>{" "}
                    {result.resnet50.prediction.toUpperCase()}
                  </p>

                  <p>
                    <strong>Ensemble:</strong>{" "}
                    {result.ensemble.prediction.toUpperCase()}
                  </p>
                </div>

                {/* INDIVIDUAL MODEL RESULTS */}

                <section className="model-comparison-section">
                  <div className="subsection-heading">
                    <h3>Individual model results</h3>
                    <span>Model comparison</span>
                  </div>

                  <div className="model-comparison-grid">
                    <ModelCard
                      title="EfficientNet-B0"
                      data={result.efficientnet}
                    />

                    <ModelCard
                      title="ResNet50"
                      data={result.resnet50}
                    />
                  </div>
                </section>

                {/* ENSEMBLE EXPLANATION */}

                <div className="explanation-box">
                  <h3>
                    <span>✳</span> Ensemble explanation
                  </h3>

                  <p>{result.explanation}</p>
                </div>

                {/* WARNING */}

                {result.warning && (
                  <p className="warning-note">
                    {result.warning}
                  </p>
                )}

                {/* RESET */}

                <button
                  className="secondary-button"
                  onClick={resetAnalysis}
                  type="button"
                >
                  ↻ Analyze Another Photo
                </button>
              </div>
            )}
          </div>
        </section>

        {/* HOW IT WORKS */}

        <section className="how-section">
          <span className="eyebrow">HOW IT WORKS</span>

          <h2>From prediction to explanation</h2>

          <div className="feature-grid">
            <article className="feature-card">
              <span className="feature-number">01</span>
              <h3>EfficientNet-B0</h3>

              <p>
                Analyzes the uploaded image and predicts whether
                it belongs to the AI-generated or real class.
              </p>
            </article>

            <article className="feature-card">
              <span className="feature-number">02</span>
              <h3>ResNet50</h3>

              <p>
                Independently analyzes the same image to provide
                a second deep learning model prediction.
              </p>
            </article>

            <article className="feature-card">
              <span className="feature-number">03</span>
              <h3>Ensemble learning</h3>

              <p>
                Combines both models' class scores using
                equal-weight soft voting.
              </p>
            </article>

            <article className="feature-card">
              <span className="feature-number">04</span>
              <h3>Grad-CAM explanation</h3>

              <p>
                Generates separate visual explanations showing
                image regions that influenced each model's prediction.
              </p>
            </article>
          </div>
        </section>

        {/* FOOTER */}

        <footer className="footer">
          <p>ImageTruth · Explainable AI Image Detection</p>
          <p>
            Model predictions are estimates, not proof of authenticity.
          </p>
        </footer>
      </main>
    </div>
  );
}

export default App;

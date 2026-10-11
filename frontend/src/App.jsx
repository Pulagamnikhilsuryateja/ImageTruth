
import { useState } from "react";
import "./App.css";

const API_URL = "http://127.0.0.1:8000/predict";

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

    if (preview) URL.revokeObjectURL(preview);

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
    if (preview) URL.revokeObjectURL(preview);

    setImage(null);
    setPreview("");
    setResult(null);
    setError("");

    const input = document.getElementById("image-upload");
    if (input) input.value = "";
  }

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
        <section className="hero">
          <span className="eyebrow">DEEP LEARNING · EXPLAINABLE AI</span>

          <h1>
            Discover the story
            <br />
            <span>behind an image.</span>
          </h1>

          <p className="hero-description">
            Analyze an image using a deep learning model and explore the
            regions that influenced its prediction with Grad-CAM.
          </p>
        </section>

        <section className="workspace">
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
                    <span>PNG, JPG, or JPEG</span>
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
                    Analyzing image...
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
              Your image is sent to your local backend for analysis.
            </p>
          </div>

          <div className="results-card">
            <div className="section-heading">
              <span className="step-number">02</span>
              <div>
                <h2>Detection results</h2>
                <p>Your model's prediction and explanation.</p>
              </div>
            </div>

            {!result && (
              <div className="empty-results">
                <div className="empty-icon">⌕</div>
                <h3>Results will appear here</h3>
                <p>
                  Upload an image and select Analyze Image to see the
                  prediction, class scores, and Grad-CAM explanation.
                </p>
              </div>
            )}

            {result && (
              <div className="result-content">
                <div
                  className={`prediction-banner ${
                    result.prediction?.toLowerCase() === "fake"
                      ? "fake"
                      : "real"
                  }`}
                >
                  <div>
                    <span className="result-overline">
                      MODEL PREDICTION
                    </span>
                    <h3>
                      {result.prediction?.toUpperCase() === "FAKE"
                        ? "AI / Fake predicted"
                        : result.prediction?.toUpperCase() === "REAL"
                          ? "Real predicted"
                          : result.prediction}
                    </h3>
                  </div>
                  <span className="prediction-symbol">
                    {result.prediction?.toLowerCase() === "fake"
                      ? "!"
                      : "✓"}
                  </span>
                </div>

                <div className="scores-section">
                  <div className="score-row">
                    <span>AI / Fake score</span>
                    <strong>{result.scores?.fake ?? "—"}%</strong>
                  </div>
                  <div className="score-track">
                    <div
                      className="score-fill fake-fill"
                      style={{
                        width: `${Math.max(
                          0,
                          Math.min(100, Number(result.scores?.fake) || 0)
                        )}%`,
                      }}
                    />
                  </div>

                  <div className="score-row">
                    <span>Real score</span>
                    <strong>{result.scores?.real ?? "—"}%</strong>
                  </div>
                  <div className="score-track">
                    <div
                      className="score-fill real-fill"
                      style={{
                        width: `${Math.max(
                          0,
                          Math.min(100, Number(result.scores?.real) || 0)
                        )}%`,
                      }}
                    />
                  </div>
                </div>

                <div className="explanation-box">
                  <h3>
                    <span>✳</span> Why did the model make this prediction?
                  </h3>
                  <p>{result.explanation}</p>
                </div>

                <div className="gradcam-section">
                  <div className="subsection-heading">
                    <h3>Visual explanation</h3>
                    <span>Grad-CAM</span>
                  </div>

                  <p className="section-description">
                    Explore the regions that influenced the model's
                    prediction. The heatmap is not proof of image origin.
                  </p>

                  <div className="gradcam-grid">
                    <div className="gradcam-item">
                      <img
                        src={`data:image/png;base64,${result.gradcam_heatmap}`}
                        alt="Grad-CAM heatmap"
                      />
                      <p>Heatmap</p>
                    </div>

                    <div className="gradcam-item">
                      <img
                        src={`data:image/png;base64,${result.gradcam_overlay}`}
                        alt="Grad-CAM heatmap overlay on original image"
                      />
                      <p>Overlay</p>
                    </div>
                  </div>
                </div>

                <section className="generator-results">
                  <h3>Possible AI Generator</h3>

                  {result.generator_analysis?.status === "success" ? (
                    <>
                      <p>
                        Overall AI-generated score:{" "}
                        {result.generator_analysis.ai_generated_score}%
                      </p>

                      {result.generator_analysis.possible_generators?.length > 0 ? (
                        <ul>
                          {result.generator_analysis.possible_generators.map(
                            (generator) => (
                              <li key={generator.name}>
                                {generator.name}: {generator.score}%
                              </li>
                            )
                          )}
                        </ul>
                      ) : (
                        <p>No reliable generator candidate was returned.</p>
                      )}

                      <small>
                        These scores are estimates, not proof of the exact app or model used.
                      </small>
                    </>
                  ) : (
                    <p>
                      {result.generator_analysis?.message ||
                        "Generator identification is currently unavailable."}
                    </p>
                  )}
                </section>

                {result.warning && (
                  <p className="warning-note">{result.warning}</p>
                )}

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

        <section className="how-section">
          <span className="eyebrow">HOW IT WORKS</span>
          <h2>From prediction to explanation</h2>

          <div className="feature-grid">
            <article className="feature-card">
              <span className="feature-number">01</span>
              <h3>Deep learning</h3>
              <p>
                EfficientNet-B0 analyzes the image and predicts one of the
                classes learned during training.
              </p>
            </article>

            <article className="feature-card">
              <span className="feature-number">02</span>
              <h3>Class scores</h3>
              <p>
                View the model's scores for the AI/Fake and Real classes.
                These scores are not necessarily calibrated probabilities.
              </p>
            </article>

            <article className="feature-card">
              <span className="feature-number">03</span>
              <h3>Visual explanation</h3>
              <p>
                Grad-CAM highlights image regions that influenced the
                selected prediction.
              </p>
            </article>
          </div>
        </section>

        <footer className="footer">
          <p>ImageTruth · Explainable AI Image Detection</p>
          <p>Model predictions are estimates, not proof of authenticity.</p>
        </footer>
      </main>
    </div>
  );
}

export default App;
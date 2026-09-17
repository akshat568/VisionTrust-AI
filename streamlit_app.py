import os
from pathlib import Path
import pandas as pd
from PIL import Image
import streamlit as st

from src.inference.pipeline import VisionTrustPipeline, InferenceResult
from src.data.cifar10 import CIFAR10_CLASSES
from scripts.download_model import download_file, DEFAULT_MODEL_URL

# Page Configuration
st.set_page_config(
    page_title="VisionTrust AI - Trustworthy Vision Inference",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = "outputs/models/baseline_resnet18_best.pth"
CENTROIDS_PATH = "outputs/metrics/train_feature_centroids.npy"
TRUST_MODEL_PATH = "outputs/models/trust_model.pkl"


@st.cache_resource(show_spinner="Loading VisionTrust ML Pipeline...")
def load_pipeline(threshold: float = 0.50) -> VisionTrustPipeline:
    """Load and cache the production VisionTrust inference pipeline.

    If the baseline ResNet-18 model checkpoint is missing locally (e.g. on Streamlit Cloud),
    it is automatically downloaded from the GitHub Release asset URL to outputs/models/.
    """
    model_path = Path(MODEL_PATH).resolve()
    centroids_path = Path(CENTROIDS_PATH).resolve()
    trust_model_path = Path(TRUST_MODEL_PATH).resolve()

    # Ensure model output directory exists
    model_path.parent.mkdir(parents=True, exist_ok=True)

    # Automatically download baseline checkpoint from GitHub Release if missing
    if not model_path.exists():
        url = os.environ.get("VISIONTRUST_MODEL_URL", DEFAULT_MODEL_URL)
        with st.spinner("Downloading baseline ResNet-18 checkpoint (~128 MB) from GitHub Release..."):
            download_file(url, model_path)

    # Validate remaining repository artifacts
    if not centroids_path.exists():
        raise FileNotFoundError(
            f"Training feature centroids artifact not found at '{centroids_path}'. "
            f"Ensure outputs/metrics/train_feature_centroids.npy is present in the repository."
        )
    if not trust_model_path.exists():
        raise FileNotFoundError(
            f"Trust model artifact not found at '{trust_model_path}'. "
            f"Ensure outputs/models/trust_model.pkl is present in the repository."
        )

    return VisionTrustPipeline(
        model_path=str(model_path),
        centroids_path=str(centroids_path),
        trust_model_path=str(trust_model_path),
        threshold=threshold,
    )


def main():
    # Sidebar
    st.sidebar.title("🛡️ VisionTrust AI")
    st.sidebar.markdown(
        """
        **Reliability-Aware Vision Classifier**
        
        Evaluates image classification outputs against model uncertainty, 
        feature distribution shift, and input image quality signals.
        """
    )

    st.sidebar.divider()
    st.sidebar.subheader("Configuration")

    threshold = st.sidebar.slider(
        "Trust Decision Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.50,
        step=0.05,
        help="Trust score threshold above which a prediction is classified as TRUSTED.",
    )

    # Artifact check
    model_exists = Path(MODEL_PATH).exists()
    centroids_exists = Path(CENTROIDS_PATH).exists()
    trust_model_exists = Path(TRUST_MODEL_PATH).exists()

    if model_exists and centroids_exists and trust_model_exists:
        st.sidebar.success("✅ Model Artifacts Loaded")
    elif not model_exists:
        st.sidebar.info("ℹ️ Model checkpoint will auto-download on first load")
    else:
        st.sidebar.error("❌ Model Artifacts Missing")

    st.sidebar.divider()
    st.sidebar.markdown(
        """
        **System Information**
        - **Classifier**: ResNet-18 (CIFAR-10)
        - **Reliability Model**: Random Forest (9 Signals)
        - **Classes**: 10 CIFAR-10 Categories
        """
    )

    # Main Page Layout
    st.title("🛡️ VisionTrust AI — Reliable Image Classification")
    st.caption(
        "Upload a CIFAR-10 image to compute model predictions alongside real-time trust scores and failure risk evaluations."
    )

    try:
        pipeline = load_pipeline(threshold=threshold)
        pipeline.threshold = threshold
    except Exception as e:
        st.error(
            f"Failed to initialize VisionTrust inference pipeline:\n\n`{e}`\n\n"
            "Please ensure `outputs/metrics/train_feature_centroids.npy` and `outputs/models/trust_model.pkl` "
            "are present in the repository."
        )
        st.stop()

    col_input, col_preview = st.columns([1, 1])

    with col_input:
        st.subheader("1. Input Image")
        uploaded_file = st.file_uploader(
            "Choose a CIFAR-10 image (PNG, JPG, JPEG)...",
            type=["png", "jpg", "jpeg"],
            help="Upload any 32x32 or standard resolution image.",
        )

        use_sample = False
        sample_path = Path("cifar_test.png")
        if uploaded_file is None and sample_path.exists():
            st.info("💡 No file uploaded. You can test with the repository sample image:")
            if st.button("Use Sample Image (cifar_test.png)"):
                use_sample = True

    image_to_process = None
    if uploaded_file is not None:
        try:
            image_to_process = Image.open(uploaded_file).convert("RGB")
        except Exception as e:
            st.error(f"Error loading uploaded image: {e}")
    elif use_sample and sample_path.exists():
        try:
            image_to_process = Image.open(sample_path).convert("RGB")
        except Exception as e:
            st.error(f"Error loading sample image: {e}")

    with col_preview:
        st.subheader("2. Preview")
        if image_to_process is not None:
            st.image(image_to_process, caption="Input Image Preview", use_container_width=True)
        else:
            st.info("Upload an image or select the sample image to view preview and run inference.")

    if image_to_process is not None:
        st.divider()
        st.subheader("3. VisionTrust Reliability Assessment")

        with st.spinner("Executing VisionTrust Inference Pipeline..."):
            result: InferenceResult = pipeline.predict_image(image_to_process)

        # Status Banner
        if result.reliability_status == "TRUSTED":
            st.success(
                f"### ✅ Status: TRUSTED\n"
                f"Prediction for **{result.predicted_class_name.upper()}** meets reliability threshold "
                f"(Trust Score: **{result.trust_score * 100:.1f}%** >= Threshold: **{threshold * 100:.1f}%**)."
            )
        else:
            st.error(
                f"### ⚠️ Status: UNTRUSTED\n"
                f"Prediction for **{result.predicted_class_name.upper()}** has high failure risk "
                f"(Trust Score: **{result.trust_score * 100:.1f}%** < Threshold: **{threshold * 100:.1f}%**). "
                f"Manual review recommended."
            )

        # Key Metrics Row
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Predicted Class", f"{result.predicted_class_name}", f"ID: {result.predicted_class_id}")
        m2.metric("Confidence", f"{result.confidence * 100:.1f}%")
        m3.metric("Trust Score", f"{result.trust_score * 100:.1f}%")
        m4.metric("Failure Prob.", f"{result.failure_probability * 100:.1f}%")
        m5.metric("Predictive Entropy", f"{result.reliability_signals['entropy']:.4f}")

        st.markdown("<br>", unsafe_allow_html=True)

        # Tabs for detailed breakdown
        tab_signals, tab_probs, tab_raw = st.tabs(
            ["📊 Reliability Features", "📈 Class Probabilities", "⚙️ Raw Inference Data"]
        )

        with tab_signals:
            st.write("#### 9-Dimensional Reliability Signal Vector")
            signals = result.reliability_signals

            sig_df = pd.DataFrame(
                [
                    {
                        "Category": "Model Uncertainty",
                        "Feature": "Confidence",
                        "Value": round(signals.get("confidence", 0.0), 4),
                        "Description": "Maximum softmax class probability",
                    },
                    {
                        "Category": "Model Uncertainty",
                        "Feature": "Entropy",
                        "Value": round(signals.get("entropy", 0.0), 4),
                        "Description": "Shannon entropy across output probabilities",
                    },
                    {
                        "Category": "Distribution Shift",
                        "Feature": "Feature Distance",
                        "Value": round(signals.get("feature_distance", 0.0), 4),
                        "Description": "Euclidean distance to predicted class training centroid",
                    },
                    {
                        "Category": "Distribution Shift",
                        "Feature": "OOD Energy Score",
                        "Value": round(signals.get("ood_score", 0.0), 4),
                        "Description": "Log-sum-exp energy-based out-of-distribution score",
                    },
                    {
                        "Category": "Model Stability",
                        "Feature": "Augmentation Consistency",
                        "Value": round(signals.get("augmentation_consistency", 0.0), 4),
                        "Description": "Agreement across test-time image augmentations",
                    },
                    {
                        "Category": "Image Quality",
                        "Feature": "Sharpness",
                        "Value": round(signals.get("sharpness", 0.0), 4),
                        "Description": "Laplacian variance blur/sharpness score",
                    },
                    {
                        "Category": "Image Quality",
                        "Feature": "Brightness",
                        "Value": round(signals.get("brightness", 0.0), 4),
                        "Description": "Normalized mean image pixel intensity",
                    },
                    {
                        "Category": "Image Quality",
                        "Feature": "Contrast",
                        "Value": round(signals.get("contrast", 0.0), 4),
                        "Description": "Standard deviation of image pixel intensities",
                    },
                    {
                        "Category": "Image Quality",
                        "Feature": "Composite Quality",
                        "Value": round(signals.get("composite_quality", 0.0), 4),
                        "Description": "Combined quality index combining sharpness, brightness, contrast",
                    },
                ]
            )

            st.dataframe(sig_df, use_container_width=True, hide_index=True)

        with tab_probs:
            st.write("#### Softmax Class Probability Distribution")
            probs_df = pd.DataFrame(
                {
                    "Class": CIFAR10_CLASSES,
                    "Probability": result.probabilities,
                }
            ).sort_values("Probability", ascending=False)

            st.bar_chart(probs_df.set_index("Class"))

        with tab_raw:
            st.write("#### Complete Inference Result Dictionary")
            st.json(result.to_dict())


if __name__ == "__main__":
    main()

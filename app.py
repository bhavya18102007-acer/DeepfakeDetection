"""
DeepShield AI - Streamlit Web Application
Interactive user interface for deepfake image and video detection,
model performance analysis, and viva demonstration.
"""

import os
import sys
import tempfile
import time
from PIL import Image
import pandas as pd
import numpy as np
import streamlit as st

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from src.utils import load_config, get_device, load_json
from src.predict_image import predict_image
from src.predict_video import predict_video
from src.preprocessing import FaceDetector

# Page configuration
st.set_page_config(
    page_title="DeepShield AI - Deepfake Detection System",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for university presentation aesthetics
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        border-radius: 8px;
        padding: 1rem;
        border-left: 4px solid #3B82F6;
        margin-bottom: 1rem;
    }
    .real-badge {
        background-color: #DEF7EC;
        color: #03543F;
        padding: 0.35rem 0.8rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 1.2rem;
        display: inline-block;
    }
    .fake-badge {
        background-color: #FDE8E8;
        color: #9B1C1C;
        padding: 0.35rem 0.8rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 1.2rem;
        display: inline-block;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_cached_face_detector():
    """Cache FaceDetector model in memory across Streamlit reloads."""
    config = load_config("configs/config.yaml")
    if config["face_detection"]["enabled"]:
        model_path = config["face_detection"]["model_path"]
        return FaceDetector(
            model_path=model_path,
            score_threshold=config["face_detection"]["score_threshold"],
            nms_threshold=config["face_detection"]["nms_threshold"],
        )
    return None


def main():
    # Load config and status
    config = load_config("configs/config.yaml")
    device = get_device(config["training"]["device"])
    checkpoint_path = config["training"]["best_checkpoint_path"]
    checkpoint_exists = os.path.exists(checkpoint_path)
    face_detector = get_cached_face_detector()

    # Sidebar
    st.sidebar.title("🛡️ DeepShield AI")
    st.sidebar.caption("Deepfake Image & Video Detection")
    st.sidebar.markdown("---")

    menu = st.sidebar.radio(
        "Navigation",
        [
            "🔍 Image Detection",
            "🎥 Video Detection",
            "📊 Evaluation & Metrics",
            "📘 Architecture & Theory",
            "❓ Viva Defense Q&A",
            "⚖️ Ethics & Privacy",
        ],
    )

    st.sidebar.markdown("---")
    st.sidebar.subheader("System Status")
    if checkpoint_exists:
        size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
        st.sidebar.success(f"Model Checkpoint: Loaded ({size_mb:.1f} MB)")
    else:
        st.sidebar.error("Model Checkpoint: Not Found")
        st.sidebar.info("Run `python scripts/smoke_test.py` or train in Colab.")

    st.sidebar.info(f"Compute Device: {str(device).upper()}")
    st.sidebar.info(f"Face Detector: {'YuNet Active' if (face_detector and face_detector.detector) else 'Center Fallback'}")

    st.sidebar.markdown("---")
    st.sidebar.caption("B.Tech AI & Data Science Capstone Project")

    # ==========================================
    # 1. IMAGE DETECTION VIEW
    # ==========================================
    if menu == "🔍 Image Detection":
        st.markdown('<div class="main-header">🔍 Deepfake Image Detection</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Upload an image containing a face to analyze facial textures and deepfake artifacts.</div>', unsafe_allow_html=True)

        if not checkpoint_exists:
            st.warning("⚠️ No trained checkpoint found at `models/best_model.pth`. Please train the model or run the smoke test.")
            return

        col1, col2 = st.columns([1, 1])

        with col1:
            uploaded_file = st.file_uploader("Choose an image (JPG, PNG, WEBP)", type=["jpg", "jpeg", "png", "webp"])
            
            # Quick sample loader buttons for live demo
            st.caption("Or test with pre-loaded demo samples:")
            sample_cols = st.columns(2)
            load_real_sample = sample_cols[0].button("Load Sample Real Image")
            load_fake_sample = sample_cols[1].button("Load Sample Fake Image")

            img_to_analyze = None
            sample_label_note = ""

            if uploaded_file is not None:
                img_to_analyze = Image.open(uploaded_file).convert("RGB")
            elif load_real_sample and os.path.exists("sample_inputs/sample_real.jpg"):
                img_to_analyze = Image.open("sample_inputs/sample_real.jpg").convert("RGB")
                sample_label_note = "(Sample Real Image Loaded)"
            elif load_fake_sample and os.path.exists("sample_inputs/sample_fake.jpg"):
                img_to_analyze = Image.open("sample_inputs/sample_fake.jpg").convert("RGB")
                sample_label_note = "(Sample Fake Image Loaded)"

            if img_to_analyze is not None:
                st.image(img_to_analyze, caption=f"Input Image {sample_label_note}", use_container_width=True)

        with col2:
            if img_to_analyze is not None:
                st.subheader("Analysis Results")
                analyze_btn = st.button("🚀 Analyze Image with DeepShield AI", type="primary")

                if analyze_btn or load_real_sample or load_fake_sample:
                    with st.spinner("Detecting face and analyzing neural features..."):
                        try:
                            res = predict_image(
                                image_input=img_to_analyze,
                                checkpoint_path=checkpoint_path,
                                config_path="configs/config.yaml",
                                face_detector=face_detector,
                            )

                            # Display verdict badge
                            verdict = res["prediction"]
                            conf = res["confidence_pct"]

                            if verdict == "REAL":
                                st.markdown(f'<div class="real-badge">VERDICT: REAL ({conf:.1f}% Confidence)</div>', unsafe_allow_html=True)
                            else:
                                st.markdown(f'<div class="fake-badge">VERDICT: FAKE / MANIPULATED ({conf:.1f}% Confidence)</div>', unsafe_allow_html=True)

                            st.write("")
                            # Probabilities
                            p_real = res["probabilities"]["REAL"] * 100
                            p_fake = res["probabilities"]["FAKE"] * 100

                            st.write(f"**P(REAL):** {p_real:.2f}%")
                            st.progress(int(p_real))

                            st.write(f"**P(FAKE):** {p_fake:.2f}%")
                            st.progress(int(p_fake))

                            st.markdown("---")
                            # Face Crop Preview
                            crop_col1, crop_col2 = st.columns([1, 1])
                            with crop_col1:
                                st.image(res["cropped_image"], caption="Extracted Face Region (20% Margin)", width=150)
                            with crop_col2:
                                st.markdown(f"**Face Detected:** {'Yes' if res['face_detected'] else 'No (Full frame analyzed)'}")
                                if res['bounding_box']:
                                    bx, by, bw, bh = res['bounding_box']
                                    st.markdown(f"**Bounding Box:** `x:{bx}, y:{by}, w:{bw}, h:{bh}`")
                                st.markdown(f"**Latency:** `{res['inference_time_ms']} ms`")
                                st.markdown(f"**Device:** `{res['device']}`")

                            st.info(f"ℹ️ {res['disclaimer']}")

                        except Exception as e:
                            st.error(f"Error analyzing image: {e}")
            else:
                st.info("👈 Upload an image or select a sample from the left panel to begin analysis.")

    # ==========================================
    # 2. VIDEO DETECTION VIEW
    # ==========================================
    elif menu == "🎥 Video Detection":
        st.markdown('<div class="main-header">🎥 Deepfake Video Detection</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Sample frames across a video timeline, detect faces, and aggregate frame-level neural predictions.</div>', unsafe_allow_html=True)

        if not checkpoint_exists:
            st.warning("⚠️ No trained checkpoint found at `models/best_model.pth`. Please train the model or run the smoke test.")
            return

        col1, col2 = st.columns([1, 1])

        with col1:
            uploaded_video = st.file_uploader("Upload video file (MP4, AVI, MOV)", type=["mp4", "avi", "mov"])
            use_demo_video = st.button("Load Pre-Generated Demo Video")

            video_path_to_process = None
            if uploaded_video is not None:
                # Save uploaded bytes to a temporary file
                temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
                temp_video.write(uploaded_video.read())
                temp_video.close()
                video_path_to_process = temp_video.name
                st.video(video_path_to_process)
            elif use_demo_video and os.path.exists("sample_inputs/sample_video.mp4"):
                video_path_to_process = "sample_inputs/sample_video.mp4"
                st.video(video_path_to_process)
                st.caption("Loaded sample demo video.")

            # Parameters
            st.markdown("#### Inference Settings")
            sample_interval = st.slider("Frame Sample Interval (Sample every N frames)", min_value=1, max_value=30, value=5)
            max_frames = st.slider("Max Frames to Process (Limits RAM usage)", min_value=5, max_value=100, value=30)
            agg_method = st.selectbox("Score Aggregation Method", ["mean", "median", "max_fake"], index=0)

        with col2:
            if video_path_to_process is not None:
                st.subheader("Video Analysis")
                analyze_vid_btn = st.button("🚀 Analyze Video Timeline", type="primary")

                if analyze_vid_btn or use_demo_video:
                    prog_bar = st.progress(0)
                    status_text = st.empty()
                    status_text.text("Extracting and analyzing video frames...")

                    try:
                        res = predict_video(
                            video_path=video_path_to_process,
                            checkpoint_path=checkpoint_path,
                            config_path="configs/config.yaml",
                            sample_interval_frames=sample_interval,
                            max_frames=max_frames,
                            aggregation_method=agg_method,
                        )

                        prog_bar.progress(100)
                        status_text.empty()

                        if res["prediction"] == "NO_FACES_DETECTED":
                            st.warning("⚠️ " + res["warning"])
                        elif res["prediction"] == "ERROR":
                            st.error(res.get("message", "Video analysis error."))
                        else:
                            verdict = res["prediction"]
                            conf = res["confidence_pct"]

                            if verdict == "REAL":
                                st.markdown(f'<div class="real-badge">OVERALL VERDICT: REAL ({conf:.1f}% Confidence)</div>', unsafe_allow_html=True)
                            else:
                                st.markdown(f'<div class="fake-badge">OVERALL VERDICT: FAKE ({conf:.1f}% Confidence)</div>', unsafe_allow_html=True)

                            st.write("")
                            # Video summary metrics
                            meta = res["video_metadata"]
                            m_col1, m_col2, m_col3 = st.columns(3)
                            m_col1.metric("Sampled Frames", res["frames_sampled"])
                            m_col2.metric("Faces Detected", res["frames_with_faces"])
                            m_col3.metric("Video Duration", f"{meta['duration_seconds']}s")

                            # Timeline visualization
                            if res["frame_results"]:
                                st.markdown("#### Frame Probability Timeline")
                                df_frames = pd.DataFrame(res["frame_results"])
                                
                                # Line chart showing P(FAKE) over time
                                chart_data = df_frames.set_index("timestamp_sec")[["prob_fake"]]
                                chart_data.columns = ["P(FAKE) Score"]
                                st.line_chart(chart_data)

                                with st.expander("View Detailed Frame-by-Frame Table"):
                                    st.dataframe(df_frames[["frame_index", "timestamp_sec", "face_detected", "prob_real", "prob_fake", "prediction"]])

                            st.info(f"ℹ️ {res['disclaimer']}")

                    except Exception as e:
                        st.error(f"Error processing video: {e}")
                    finally:
                        # Clean up temp file if created
                        if uploaded_video is not None and os.path.exists(video_path_to_process):
                            try:
                                os.remove(video_path_to_process)
                            except Exception:
                                pass
            else:
                st.info("👈 Upload a video file or load the demo video to begin analysis.")

    # ==========================================
    # 3. EVALUATION & METRICS VIEW
    # ==========================================
    elif menu == "📊 Evaluation & Metrics":
        st.markdown('<div class="main-header">📊 Model Evaluation & Experimental Results</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Objective performance measurements computed on the held-out test split.</div>', unsafe_allow_html=True)

        eval_json_path = "outputs/evaluation_results.json"
        if os.path.exists(eval_json_path):
            eval_data = load_json(eval_json_path)

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Test Accuracy", f"{eval_data['accuracy']*100:.2f}%")
            m2.metric("Fake Precision", f"{eval_data['precision_fake']:.4f}")
            m3.metric("Fake Recall", f"{eval_data['recall_fake']:.4f}")
            m4.metric("Fake F1-Score", f"{eval_data['f1_fake']:.4f}")

            st.markdown("---")
            g_col1, g_col2 = st.columns(2)

            with g_col1:
                st.subheader("Confusion Matrix")
                if os.path.exists("outputs/confusion_matrix.png"):
                    st.image("outputs/confusion_matrix.png", use_container_width=True)
                else:
                    st.info("Confusion matrix plot not found.")

            with g_col2:
                st.subheader("ROC Curve")
                if os.path.exists("outputs/roc_curve.png"):
                    st.image("outputs/roc_curve.png", use_container_width=True)
                else:
                    st.info("ROC plot not found.")

            st.markdown("---")
            c_col1, c_col2 = st.columns(2)

            with c_col1:
                st.subheader("Training & Validation Loss")
                if os.path.exists("outputs/loss_curve.png"):
                    st.image("outputs/loss_curve.png", use_container_width=True)
                else:
                    st.info("Loss curve not generated yet.")

            with c_col2:
                st.subheader("Accuracy Progression")
                if os.path.exists("outputs/accuracy_curve.png"):
                    st.image("outputs/accuracy_curve.png", use_container_width=True)
                else:
                    st.info("Accuracy curve not generated yet.")

            # Classification Report Text
            if os.path.exists("outputs/classification_report.txt"):
                with st.expander("View Full Scikit-Learn Classification Report"):
                    with open("outputs/classification_report.txt", "r") as f:
                        st.code(f.read())
        else:
            st.info("No evaluation results found in `outputs/`. Run `python src/evaluate.py` or `python scripts/smoke_test.py` to generate metrics.")

    # ==========================================
    # 4. ARCHITECTURE & THEORY VIEW
    # ==========================================
    elif menu == "📘 Architecture & Theory":
        st.markdown('<div class="main-header">📘 System Architecture & Methodology</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Technical design of DeepShield AI for university presentation and viva defense.</div>', unsafe_allow_html=True)

        st.markdown("""
        ### 1. Convolutional Neural Networks & ResNet-18
        - **What is a CNN?** A deep neural network utilizing learnable convolution filters that slide over image pixels to detect local patterns (edges, textures, shapes).
        - **Why ResNet-18?** Deeper neural networks often suffer from the **vanishing gradient problem**, where gradient signals diminish to near zero as they propagate back through layers. **ResNet (Residual Network)** introduces **skip / residual connections** ($y = F(x) + x$) that allow gradients to flow directly through identity shortcuts.
        - ResNet-18 provides 18 convolutional and pooling layers with ~11.7 million parameters, offering an optimal balance of fast inference on student laptops (8 GB RAM) and high representation capacity.

        ### 2. Transfer Learning Strategy
        - Pretrained weights initialized on **ImageNet-1K** provide general visual representations (edge detection, color distributions, face geometry).
        - The final classification layer is replaced with a custom binary classifier head:
          ```python
          model.fc = nn.Sequential(
              nn.Dropout(p=0.3),
              nn.Linear(512, 2)  # Output: [REAL, FAKE]
          )
          ```
        - Fine-tuning the network enables the model to detect subtle manipulation cues (warping artifacts, blending boundaries, and chromatic inconsistencies).

        ### 3. Face Detection & Margin Expansion
        - Deepfake manipulation artifacts (blending seams, resolution mismatches) are concentrated along facial boundaries and jawlines.
        - DeepShield AI uses **OpenCV YuNet**, an ONNX-based deep learning face detector.
        - A **20% margin** is added around each detected bounding box to ensure facial contours and boundary artifacts are preserved rather than clipped.

        ### 4. Video Frame Sampling & Statistical Aggregation
        - Videos are processed as sequences of frames sampled at regular intervals (e.g. every 10 frames).
        - Frame scores are aggregated using **mean probability** or **median pooling**:
          $$\\bar{P}(\\text{FAKE}) = \\frac{1}{N} \\sum_{i=1}^{N} P_i(\\text{FAKE})$$
        - This prevents single-frame noise or motion blur from falsely triggering a deepfake verdict.

        ### 5. Video-Level Data Leakage Prevention
        - **Crucial Rule:** Source videos are split into `train`, `val`, and `test` **before** extracting frames.
        - Frames from the same source video (and associated manipulation pairs) are strictly kept within the same split to prevent the model from memorizing individual identities rather than deepfake artifacts.
        """)

    # ==========================================
    # 5. VIVA DEFENSE Q&A VIEW
    # ==========================================
    elif menu == "❓ Viva Defense Q&A":
        st.markdown('<div class="main-header">❓ Viva Voce & Oral Defense Preparation</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">40+ comprehensive questions and answers categorized for your examination committee.</div>', unsafe_allow_html=True)

        st.markdown("Detailed viva questions are also available in `docs/viva_questions.md`.")

        with st.expander("Q1: What is a deepfake and how does it differ from traditional photo editing?"):
            st.write("**Answer:** A deepfake is synthetic media (image or video) created or altered using deep-learning generative models (such as Autoencoders, GANs, or Diffusion models) to replace a person's likeness, voice, or expressions. Unlike manual Photoshop editing, deepfakes are generated algorithmically by learning latent representations of faces, allowing automated manipulation across video sequences.")

        with st.expander("Q2: Why did you choose ResNet-18 instead of a larger model like ResNet-50 or ViT?"):
            st.write("**Answer:** ResNet-18 has approximately 11.7 million parameters and requires around 45 MB of memory. For a university capstone project running on standard student laptops with 8 GB RAM, ResNet-18 offers real-time inference latency (<50 ms per image on CPU), prevents GPU out-of-memory errors, and achieves competitive accuracy without overfitting on modest academic dataset subsets.")

        with st.expander("Q3: What is the purpose of residual skip connections in ResNet?"):
            st.write("**Answer:** In traditional deep networks, repeated matrix multiplications cause gradients to shrink exponentially during backpropagation (vanishing gradients). Residual connections introduce shortcut paths $y = F(x) + x$, allowing gradients to flow directly backwards to earlier layers without vanishing, enabling stable training of deeper architectures.")

        with st.expander("Q4: What is data leakage and how did your project prevent it in video datasets?"):
            st.write("**Answer:** Data leakage occurs when test data information inadvertently influences the training set. In video deepfake detection, if frames from the same video are randomly distributed across train and test sets, the model learns to recognize the person's face rather than deepfake manipulation. DeepShield AI prevents this by splitting datasets strictly at the source video level: all frames from a given video belong entirely to train, val, or test.")

        with st.expander("Q5: What are False Positives and False Negatives in deepfake detection, and which is more severe?"):
            st.write("**Answer:** A False Positive (FP) occurs when genuine media is incorrectly classified as FAKE. A False Negative (FN) occurs when a deepfake is missed and classified as REAL. In journalism and legal contexts, a False Positive harms innocent individuals by falsely discrediting genuine recordings. In national security and elections, a False Negative allows deceptive propaganda to spread unchecked. Both must be balanced using precision and recall metrics.")

    # ==========================================
    # 6. ETHICS & PRIVACY VIEW
    # ==========================================
    elif menu == "⚖️ Ethics & Privacy":
        st.markdown('<div class="main-header">⚖️ Ethics, Privacy, and Responsible AI</div>', unsafe_allow_html=True)
        st.markdown("""
        ### Privacy Protection
        - **Local Processing:** All uploaded media is processed entirely in local application memory or temporary files on the user's host machine.
        - **No Third-Party Transmission:** No images or videos are transmitted to external APIs or cloud servers.
        - **Automatic Resource Cleanup:** Temporary files created during video frame extraction are purged immediately upon analysis completion.

        ### Domain Shift & Real-World Limitations
        - **Domain Shift:** Deep learning classifiers often experience performance degradation when applied to media generated by unseen manipulation algorithms, camera sensors, lighting conditions, or compression codecs not present in the training set.
        - **Not Legal Proof:** DeepShield AI's output represents an empirical statistical estimate under the learned model parameters, NOT definitive legal proof of authenticity.
        - Decisions regarding media authenticity in sensitive contexts (court of law, journalism, public integrity) should incorporate multi-factor forensic verification.
        """)


if __name__ == "__main__":
    main()

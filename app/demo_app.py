"""Interactive Streamlit Demonstration Interface.

Allows users to upload or select Indic audio samples, simulate lossy compression,
tune operating decision thresholds, and observe real-time predictions and
spectrogram features from Model A (SVM), Model B1 (CNN), Model B2 (LCNN-MFM),
Model B3 (SE-ResNet), and the Hybrid Forensic Ensemble.
"""

import os
import io
import torch
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
import streamlit as st

from src.preprocessing.audio_loader import load_and_standardize_audio
from src.features.extractors import FeatureExtractor
from src.features.classical_features import ClassicalFeatureExtractor
from src.models.lightweight_cnn import LightweightAudioCNN
from src.models.classical_baseline import ClassicalBaseline
from src.models.lcnn import LCNN
from src.models.se_resnet import SEResNet
from src.models.distilhubert_classifier import DistilHuBERTClassifier
from src.models.ensemble import ForensicEnsemble
from src.robustness.ffmpeg_codec import AudioCodecSimulator


st.set_page_config(page_title="Indic Deepfake Speech Detector", layout="wide")


@st.cache_resource
def load_models():
    device = torch.device("cpu")
    extractor = FeatureExtractor(sample_rate=16000, n_mels=80)
    classical_extractor = ClassicalFeatureExtractor(sample_rate=16000, n_coefficients=30)
    codec_sim = AudioCodecSimulator()

    # 1. Model B1: Clean 2D-CNN
    cnn_clean = LightweightAudioCNN(base_channels=16, fc_hidden=64)
    if os.path.exists("checkpoints/cnn_clean_best.pt"):
        ckpt = torch.load("checkpoints/cnn_clean_best.pt", map_location=device)
        cnn_clean.load_state_dict(ckpt["model_state_dict"])
    cnn_clean.eval()

    # 2. Model B2: LCNN-MFM
    lcnn_model = LCNN(in_channels=1, dropout_rate=0.3)
    if os.path.exists("checkpoints/lcnn_mfm_augmented_best.pt"):
        ckpt = torch.load("checkpoints/lcnn_mfm_augmented_best.pt", map_location=device)
        lcnn_model.load_state_dict(ckpt["model_state_dict"])
    lcnn_model.eval()

    # 3. Model B3: SE-ResNet
    se_resnet = SEResNet(in_channels=1, base_channels=24, dropout_rate=0.3)
    if os.path.exists("checkpoints/se_resnet_augmented_best.pt"):
        ckpt = torch.load("checkpoints/se_resnet_augmented_best.pt", map_location=device)
        se_resnet.load_state_dict(ckpt["model_state_dict"])
    se_resnet.eval()

    # 4. Model A: SVM
    svm_model = ClassicalBaseline(model_type="svm_rbf")
    if os.path.exists("checkpoints/model_a_svm_rbf.joblib"):
        svm_model.load("checkpoints/model_a_svm_rbf.joblib")

    # 5. Model B4: DistilHuBERT (SOTA Pretrained Transformer)
    distilhubert_model = None
    if os.path.exists("checkpoints/distilhubert_best.pt"):
        try:
            distilhubert_model = DistilHuBERTClassifier()
            ckpt = torch.load("checkpoints/distilhubert_best.pt", map_location=device)
            distilhubert_model.load_state_dict(ckpt["model_state_dict"])
            distilhubert_model.eval()
        except Exception as e:
            st.warning(f"Note: DistilHuBERT weights could not be loaded: {e}")

    # 6. Hybrid Ensemble
    ensemble = ForensicEnsemble(weights={"svm": 0.34, "lcnn": 0.33, "se_resnet": 0.33})

    return extractor, classical_extractor, codec_sim, cnn_clean, lcnn_model, se_resnet, svm_model, distilhubert_model, ensemble


def main():
    st.title("🎙️ Cross-Lingual & Codec-Robust Speech Deepfake Detector")
    st.markdown(
        "**Multi-Architecture Anti-Spoofing Suite:** Dual Held-Out Evaluation (Indo-Aryan $\\rightarrow$ Dravidian & Tibeto-Burman) "
        "featuring **LCNN-MFM**, **SE-ResNet**, and **Hybrid Forensic Ensembling**."
    )

    extractor, classical_extractor, codec_sim, cnn_clean, lcnn_model, se_resnet, svm_model, distilhubert_model, ensemble = load_models()

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("1. Audio Input & Transmission Setup")
        uploaded_file = st.file_uploader("Upload Audio (WAV / MP3 / Opus)", type=["wav", "mp3", "opus"])

        codec_option = st.selectbox(
            "Transmission Codec Simulation:",
            ["Clean (PCM 16kHz)", "MP3 @ 64 kbps", "MP3 @ 32 kbps", "Opus @ 24 kbps", "Opus @ 16 kbps"],
        )

        codec_map = {
            "Clean (PCM 16kHz)": "clean",
            "MP3 @ 64 kbps": "mp3_64k",
            "MP3 @ 32 kbps": "mp3_32k",
            "Opus @ 24 kbps": "opus_24k",
            "Opus @ 16 kbps": "opus_16k",
        }

        lang_group = st.selectbox(
            "Linguistic Evaluation Category:",
            [
                "Seen Indo-Aryan (Hindi, Marathi, Bengali, Gujarati, etc.)",
                "Held-Out Target 1: Dravidian (Tamil, Telugu, Kannada, Malayalam)",
                "Held-Out Target 2: Tibeto-Burman (Bodo, Manipuri / Meitei)",
            ],
        )

        threshold = st.slider("Forensic Decision Threshold (tau):", min_value=0.10, max_value=0.90, value=0.50, step=0.02)
        st.caption("Tip: Use tau = 0.50 for default balance, or tune lower/higher to favor Recall or Precision.")

        if uploaded_file is not None:
            audio_bytes = uploaded_file.read()
            audio_clean = load_and_standardize_audio(audio_bytes, target_sr=16000, duration=3.0)

            # Apply compression
            selected_codec = codec_map[codec_option]
            if selected_codec == "clean":
                audio_eval = audio_clean
            else:
                audio_eval = codec_sim.compress_decompress(audio_clean, sample_rate=16000, codec_profile=selected_codec)

            st.audio(audio_eval, sample_rate=16000)

    with col_right:
        st.subheader("2. Multi-Model Forensic Predictions")

        if uploaded_file is not None:
            spec = extractor.extract_log_mel(audio_eval)
            feat_1d = classical_extractor.extract_vector(audio_eval).reshape(1, -1)
            t_spec = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0).float()

            # Predictions
            prob_svm = float(svm_model.predict_proba(feat_1d)[0, 1])
            prob_cnn = float(cnn_clean.predict_proba(t_spec)[0, 0])
            prob_lcnn = float(lcnn_model.predict_proba(t_spec)[0, 0])
            prob_se = float(se_resnet.predict_proba(t_spec)[0, 0])

            # Ensemble prediction
            prob_ens = float(ensemble.predict_proba({
                "svm": np.array([prob_svm]),
                "lcnn": np.array([prob_lcnn]),
                "se_resnet": np.array([prob_se]),
            })[0])

            # DistilHuBERT prediction
            prob_hubert = None
            if distilhubert_model is not None:
                raw_t = torch.from_numpy(audio_eval).unsqueeze(0).float()
                with torch.no_grad():
                    prob_hubert = float(distilhubert_model.predict_proba(raw_t)[0, 0])

            st.markdown("### Multi-Model Forensic Probabilities")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Model A (LFCC RBF-SVM)", f"{prob_svm*100:.1f}% Fake")
                st.metric("Model B1 (Baseline 2D-CNN)", f"{prob_cnn*100:.1f}% Fake")
            with c2:
                st.metric("Model B2 (LCNN with MFM)", f"{prob_lcnn*100:.1f}% Fake")
                st.metric("Model B3 (SE-ResNet-18)", f"{prob_se*100:.1f}% Fake")
            with c3:
                if prob_hubert is not None:
                    st.metric("🔥 Model B4 (DistilHuBERT SSL)", f"{prob_hubert*100:.1f}% Fake")
                st.metric("🌟 Hybrid Ensemble", f"{prob_ens*100:.1f}% Fake")

            st.markdown("---")
            verdict_source = st.radio(
                "Select Active Classifier for Primary Verdict:",
                ["DistilHuBERT (Empirical SOTA)", "Hybrid Ensemble", "SE-ResNet-18", "LCNN-MFM"],
                index=0 if prob_hubert is not None else 1,
                horizontal=True
            )

            if verdict_source == "DistilHuBERT (Empirical SOTA)" and prob_hubert is not None:
                active_prob = prob_hubert
                active_name = "DistilHuBERT"
            elif verdict_source == "SE-ResNet-18":
                active_prob = prob_se
                active_name = "SE-ResNet-18"
            elif verdict_source == "LCNN-MFM":
                active_prob = prob_lcnn
                active_name = "LCNN-MFM"
            else:
                active_prob = prob_ens
                active_name = "Hybrid Ensemble"

            # Final verdict based on active model & selected threshold
            if active_prob >= threshold:
                st.error(f"🚨 **VERDICT: AI-GENERATED DEEPFAKE** ({active_name} Confidence: {active_prob*100:.1f}%, tau={threshold:.2f})")
            else:
                st.success(f"✅ **VERDICT: BONAFIDE HUMAN SPEECH** ({active_name} Confidence: {(1.0-active_prob)*100:.1f}%, tau={threshold:.2f})")

            # Spectrogram visualizer
            st.markdown("### Log-Mel Spectrogram")
            fig, ax = plt.subplots(figsize=(8, 3))
            cax = ax.imshow(spec, origin="lower", aspect="auto", cmap="magma")
            ax.set_title("Log-Mel Spectrogram (80 Mel Bands, 16 kHz)")
            ax.set_xlabel("Time Frames")
            ax.set_ylabel("Mel Frequency Bins")
            fig.colorbar(cax, ax=ax, format="%+2.0f dB")
            st.pyplot(fig)
        else:
            st.info("Upload an audio file on the left panel to begin forensic analysis.")


if __name__ == "__main__":
    main()

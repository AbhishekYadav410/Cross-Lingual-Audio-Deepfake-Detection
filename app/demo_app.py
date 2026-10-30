"""Interactive Streamlit Demonstration Interface.

Allows users to upload or select Indic audio samples, simulate lossy compression,
tune operating decision thresholds, and observe real-time predictions and
spectrogram features from Model A (SVM), Model B1 (CNN), Model B2 (LCNN-MFM),
Model B3 (SE-ResNet), and the Hybrid Forensic Ensemble.
"""

import os
import sys

# Ensure repository root is on sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import io
import torch
import numpy as np
import pandas as pd
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

    ckpt_dir = os.path.join(PROJECT_ROOT, "checkpoints")

    # 1. Model B1: Clean 2D-CNN
    cnn_clean = LightweightAudioCNN(base_channels=16, fc_hidden=64)
    cnn_clean_path = os.path.join(ckpt_dir, "cnn_clean_best.pt")
    if os.path.exists(cnn_clean_path):
        ckpt = torch.load(cnn_clean_path, map_location=device)
        cnn_clean.load_state_dict(ckpt["model_state_dict"])
    cnn_clean.eval()

    # 2. Model B2: LCNN-MFM
    lcnn_model = LCNN(in_channels=1, dropout_rate=0.3)
    lcnn_path = os.path.join(ckpt_dir, "lcnn_mfm_augmented_best.pt")
    if os.path.exists(lcnn_path):
        ckpt = torch.load(lcnn_path, map_location=device)
        lcnn_model.load_state_dict(ckpt["model_state_dict"])
    lcnn_model.eval()

    # 3. Model B3: SE-ResNet
    se_resnet = SEResNet(in_channels=1, base_channels=24, dropout_rate=0.3)
    se_resnet_path = os.path.join(ckpt_dir, "se_resnet_augmented_best.pt")
    if os.path.exists(se_resnet_path):
        ckpt = torch.load(se_resnet_path, map_location=device)
        se_resnet.load_state_dict(ckpt["model_state_dict"])
    se_resnet.eval()

    # 4. Model A: SVM
    svm_model = ClassicalBaseline(model_type="svm_rbf")
    svm_path = os.path.join(ckpt_dir, "model_a_svm_rbf.joblib")
    if os.path.exists(svm_path):
        svm_model.load(svm_path)

    # 5. Model B4: DistilHuBERT (SOTA Pretrained Transformer)
    distilhubert_model = None
    distilhubert_path = os.path.join(ckpt_dir, "distilhubert_best.pt")
    if os.path.exists(distilhubert_path):
        try:
            distilhubert_model = DistilHuBERTClassifier()
            ckpt = torch.load(distilhubert_path, map_location=device)
            state_dict = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
            distilhubert_model.load_state_dict(state_dict)
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

            # Model data collection
            models_data = []

            # 1. DistilHuBERT (Empirical SOTA)
            if distilhubert_model is not None and prob_hubert is not None:
                models_data.append({
                    "name": "Model B4 (DistilHuBERT SSL)",
                    "accuracy": 96.53,
                    "auroc": 0.9943,
                    "eer": 3.47,
                    "prob": prob_hubert,
                    "vote": "Fake" if prob_hubert >= threshold else "Bonafide",
                    "eligible": True,
                    "quality": "🥇 SOTA (Best Quality)",
                    "status": "✅ Included in Majority Vote",
                })

            # 2. SE-ResNet-18
            models_data.append({
                "name": "Model B3 (SE-ResNet-18)",
                "accuracy": 90.28,
                "auroc": 0.9655,
                "eer": 10.71,
                "prob": prob_se,
                "vote": "Fake" if prob_se >= threshold else "Bonafide",
                "eligible": True,
                "quality": "🥈 High Quality",
                "status": "✅ Included in Majority Vote",
            })

            # 3. LCNN-MFM
            models_data.append({
                "name": "Model B2 (LCNN with MFM)",
                "accuracy": 85.76,
                "auroc": 0.9509,
                "eer": 12.84,
                "prob": prob_lcnn,
                "vote": "Fake" if prob_lcnn >= threshold else "Bonafide",
                "eligible": True,
                "quality": "🥉 High Quality",
                "status": "✅ Included in Majority Vote",
            })

            # 4. Model A (LFCC RBF-SVM)
            models_data.append({
                "name": "Model A (LFCC RBF-SVM)",
                "accuracy": 82.64,
                "auroc": 0.9269,
                "eer": 16.22,
                "prob": prob_svm,
                "vote": "Fake" if prob_svm >= threshold else "Bonafide",
                "eligible": True,
                "quality": "Moderate Quality",
                "status": "✅ Included in Majority Vote",
            })

            # 5. Model B1 (Baseline 2D-CNN) - RULED OUT due to lower accuracy metrics
            models_data.append({
                "name": "Model B1 (Baseline 2D-CNN)",
                "accuracy": 73.61,
                "auroc": 0.8124,
                "eer": 26.39,
                "prob": prob_cnn,
                "vote": "Fake" if prob_cnn >= threshold else "Bonafide",
                "eligible": False,
                "quality": "⚠️ Low Quality",
                "status": "❌ Ruled Out (Inferior Accuracy Metric)",
            })

            # Majority voting among eligible models
            eligible_models = [m for m in models_data if m["eligible"]]
            fake_votes = sum(1 for m in eligible_models if m["vote"] == "Fake")
            real_votes = sum(1 for m in eligible_models if m["vote"] == "Bonafide")
            total_votes = len(eligible_models)

            # Determine best model by benchmark accuracy
            best_model = max(eligible_models, key=lambda m: m["accuracy"])

            # Final verdict determination (majority vote, tie-broken by best model)
            if fake_votes > real_votes:
                is_deepfake = True
            elif real_votes > fake_votes:
                is_deepfake = False
            else:
                is_deepfake = (best_model["vote"] == "Fake")

            # Final Verdict Display
            if is_deepfake:
                st.error(
                    f"🚨 **FINAL VERDICT: AI-GENERATED DEEPFAKE**\n\n"
                    f"• **Majority Decision:** **{fake_votes} of {total_votes} qualified models** voted **Deepfake** (threshold $\\tau = {threshold:.2f}$)\n\n"
                    f"• **Best Model Result ({best_model['name']}):** **{best_model['prob']*100:.1f}% Fake** (Benchmark Accuracy: {best_model['accuracy']:.1f}%, AUROC: {best_model['auroc']:.4f})\n\n"
                    f"• *Rule-out Policy: Model B1 (Baseline 2D-CNN) was excluded from the verdict due to inferior accuracy metrics ({models_data[-1]['accuracy']:.1f}% Acc, {models_data[-1]['eer']:.1f}% EER).*"
                )
            else:
                st.success(
                    f"✅ **FINAL VERDICT: BONAFIDE HUMAN SPEECH**\n\n"
                    f"• **Majority Decision:** **{real_votes} of {total_votes} qualified models** voted **Bonafide Human** (threshold $\\tau = {threshold:.2f}$)\n\n"
                    f"• **Best Model Result ({best_model['name']}):** **{(1.0 - best_model['prob'])*100:.1f}% Bonafide** (Benchmark Accuracy: {best_model['accuracy']:.1f}%, AUROC: {best_model['auroc']:.4f})\n\n"
                    f"• *Rule-out Policy: Model B1 (Baseline 2D-CNN) was excluded from the verdict due to inferior accuracy metrics ({models_data[-1]['accuracy']:.1f}% Acc, {models_data[-1]['eer']:.1f}% EER).*"
                )

            # Best Model Result Highlight Card
            st.markdown("### 🏆 Best Model Result")
            st.info(
                f"**Top Performing Architecture:** **{best_model['name']}** ({best_model['quality']})\n\n"
                f"• **Sample Prediction:** **{best_model['prob']*100:.1f}% Fake** ({(1.0 - best_model['prob'])*100:.1f}% Bonafide)\n\n"
                f"• **Benchmark Quality:** **{best_model['accuracy']:.2f}% Accuracy** | **{best_model['auroc']:.4f} AUROC** | **{best_model['eer']:.2f}% EER**"
            )

            # Comparative Analysis & Quality Breakdown Table
            st.markdown("### 📊 Model Results & Quality Comparison")
            df_comp = pd.DataFrame([
                {
                    "Model": m["name"],
                    "Quality Rank": m["quality"],
                    "Benchmark Acc": f"{m['accuracy']:.1f}%",
                    "Benchmark AUROC": f"{m['auroc']:.3f}",
                    "Sample Score": f"{m['prob']*100:.1f}% Fake",
                    "Model Vote": "🚨 Fake" if m["vote"] == "Fake" else "✅ Bonafide",
                    "Voting Status": m["status"],
                }
                for m in models_data
            ])
            st.dataframe(df_comp, use_container_width=True, hide_index=True)

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

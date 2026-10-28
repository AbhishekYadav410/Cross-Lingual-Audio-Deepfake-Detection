# Cross-Lingual Generalization and Codec Robustness of AI-Generated Speech Detectors for Indian Languages

[![Python 3.12](https://img.shields.io/badge/Python-3.12-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch 2.6](https://img.shields.io/badge/PyTorch-2.6%20CUDA%2012.4-EE4C2C.svg?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-ntu--spml%2Fdistilhubert-FFD21E.svg?style=flat&logo=huggingface&logoColor=black)](https://huggingface.co/ntu-spml/distilhubert)
[![Streamlit Demo](https://img.shields.io/badge/Streamlit-Demo%20App-FF4B4B.svg?style=flat&logo=streamlit&logoColor=white)](app/demo_app.py)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An academic research suite and reproducible benchmark investigating **cross-lingual transferability**, **lossy codec robustness**, and **out-of-distribution generalization** of lightweight and self-supervised audio anti-spoofing models across **16 Indian languages** (Indo-Aryan, Dravidian, and Tibeto-Burman families), as well as international benchmarks (**ASVspoof 5** and **MLAAD**).

---

## 📌 Research Overview & Primary Research Questions

### RQ1 — Cross-Lingual Generalization
> *How well do speech deepfake detectors trained exclusively on Indo-Aryan languages generalize to completely held-out Indian language families (Dravidian and Tibeto-Burman)?*

* **Seen Family (Indo-Aryan & Contact):** Hindi, Bengali, Marathi, Gujarati, Odia, Bhojpuri, Chhattisgarhi, Maithili, Magahi.
* **Held-Out Target 1 (Dravidian):** Tamil, Telugu, Kannada, Malayalam.
* **Held-Out Target 2 (Tibeto-Burman):** Bodo, Manipuri / Meitei.

### RQ2 — Codec Robustness Under Cellular Compression
> *How does progressive lossy audio compression degrade detection boundaries, and does data augmentation or pre-trained speech representation restore robustness under cellular bitrates?*

* **Tested Codec Profiles:** Clean (PCM 16 kHz), MP3 @ 64 kbps, MP3 @ 32 kbps, Opus @ 24 kbps, and extreme cellular Opus @ 16 kbps.

### RQ3 — Accuracy vs. Efficiency Pareto Frontier
> *What trade-offs exist between detection performance, parameter footprint, checkpoint size, and inference latency across classical, compact CNN, and self-supervised architectures?*

---

## 🏆 Key Empirical Findings

1. **Self-Supervised Representations (DistilHuBERT) Are SOTA:**
   - **Dravidian Generalization:** Achieves **0.9959 AUROC**, **3.01% EER**, **94.88% Accuracy**, **98.47% Recall**, and **91.71% Precision**—slashing error rates by **$>6\times$** compared to LCNN-MFM (18.07% EER) and **$>3\times$** compared to SE-ResNet-18 (9.67% EER).
   - **Tibeto-Burman Generalization:** Achieves **0.9335 AUROC**, **10.53% EER**, and **98.75% Recall** (detecting 237 of 240 synthetic samples).
2. **Near-Zero Codec Degradation:**
   - DistilHuBERT drops by merely **$-0.0022$ AUROC** from pristine audio ($0.9943$) down to extreme Opus 16 kbps ($0.9921$), exhibiting intrinsic codec invariance.
3. **Out-of-Distribution Transfer:**
   - **ASVspoof 5:** DistilHuBERT scores **0.6995 AUROC** and **32.33% EER** ($+11.2\%$ absolute gain over compact CNNs).
   - **MLAAD:** All models drop to near-chance ($0.41 - 0.57$ AUROC), confirming the open challenge of vocoder/synthesis artifact overfitting documented by Müller et al. (Interspeech 2022).
4. **Pareto Frontier:**
   - **DistilHuBERT (23.9M params, 95.6 MB, 42.5 ms latency):** Recommended for cloud/server-side forensic audits.
   - **SE-ResNet-18 (365k params, 4.26 MB, 9.9 ms latency):** Recommended for edge/mobile devices.
   - **LCNN-MFM (174k params, 2.03 MB, 6.7 ms latency):** Recommended for embedded/DSP hardware.

---

## 📊 Comprehensive Benchmark Results

Evaluated with **500-resample Bootstrap 95% Confidence Intervals**:

| Architecture | Evaluation Split | AUROC [95% CI] | EER (%) | Accuracy (%) | Precision (%) | Recall (%) | F1 (%) | Latency (CPU) | Size (MB) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DistilHuBERT (SSL)** | **Seen Indo-Aryan** | **0.9943 [0.986, 1.000]** | **3.47%** | **96.53%** | **94.52%** | **98.57%** | **96.50%** | 42.5 ms | 95.6 MB |
| Hybrid Ensemble | Seen Indo-Aryan | 0.9847 [0.973, 0.993] | 7.14% | 92.36% | 87.82% | 97.86% | 92.36% | 23.8 ms | 7.87 MB |
| SE-ResNet-18 | Seen Indo-Aryan | 0.9655 [0.942, 0.983] | 10.71% | 90.28% | 87.33% | 93.57% | 90.28% | 9.9 ms | 4.26 MB |
| LCNN-MFM | Seen Indo-Aryan | 0.9509 [0.924, 0.971] | 12.84% | 85.76% | 83.67% | 87.86% | 85.76% | 6.7 ms | 2.03 MB |
| Calibrated RBF-SVM | Seen Indo-Aryan | 0.9269 [0.900, 0.952] | 16.22% | 82.64% | 86.29% | 76.43% | 82.52% | 7.2 ms | 1.58 MB |
| Baseline 2D-CNN | Seen Indo-Aryan | 0.8124 [0.760, 0.862] | 26.39% | 73.61% | 72.15% | 75.71% | 73.59% | 3.5 ms | 1.23 MB |
| **DistilHuBERT (SSL)** | **Held-Out 1 (Dravidian)** | **0.9959 [0.994, 0.998]** | **3.01%** | **94.88%** | **91.71%** | **98.47%** | **94.97%** | 42.5 ms | 95.6 MB |
| Hybrid Ensemble | Held-Out 1 (Dravidian) | 0.9752 [0.966, 0.983] | 8.11% | 87.12% | 80.72% | 96.95% | 87.04% | 23.8 ms | 7.87 MB |
| SE-ResNet-18 | Held-Out 1 (Dravidian) | 0.9684 [0.958, 0.978] | 9.67% | 89.00% | 83.81% | 96.18% | 88.97% | 9.9 ms | 4.26 MB |
| LCNN-MFM | Held-Out 1 (Dravidian) | 0.9082 [0.888, 0.927] | 18.07% | 80.37% | 76.94% | 85.75% | 80.35% | 6.7 ms | 2.03 MB |
| Calibrated RBF-SVM | Held-Out 1 (Dravidian) | 0.9092 [0.889, 0.927] | 17.20% | 81.63% | 77.95% | 87.28% | 81.59% | 7.2 ms | 1.58 MB |
| **DistilHuBERT (SSL)** | **Held-Out 2 (Tibeto-Burman)** | **0.9335 [0.908, 0.956]** | **10.53%** | **83.58%** | **75.96%** | **98.75%** | **85.87%** | 42.5 ms | 95.6 MB |
| Hybrid Ensemble | Held-Out 2 (Tibeto-Burman) | 0.9464 [0.927, 0.965] | 13.19% | 83.37% | 77.10% | 95.42% | 83.08% | 23.8 ms | 7.87 MB |
| SE-ResNet-18 | Held-Out 2 (Tibeto-Burman) | 0.9314 [0.908, 0.953] | 14.17% | 83.58% | 77.36% | 95.42% | 83.30% | 9.9 ms | 4.26 MB |
| LCNN-MFM | Held-Out 2 (Tibeto-Burman) | 0.8659 [0.834, 0.898] | 22.13% | 79.79% | 78.12% | 83.33% | 79.75% | 6.7 ms | 2.03 MB |
| Calibrated RBF-SVM | Held-Out 2 (Tibeto-Burman) | 0.8540 [0.820, 0.887] | 21.28% | 78.74% | 76.43% | 83.75% | 78.66% | 7.2 ms | 1.58 MB |
| **DistilHuBERT (SSL)** | **ASVspoof 5 Benchmark** | **0.6995 [0.637, 0.756]** | **32.33%** | **67.67%** | **67.69%** | **66.67%** | **67.64%** | 42.5 ms | 95.6 MB |
| LCNN-MFM | ASVspoof 5 Benchmark | 0.5873 [0.522, 0.650] | 44.00% | 57.00% | 55.45% | 70.00% | 56.57% | 6.7 ms | 2.03 MB |
| SE-ResNet-18 | ASVspoof 5 Benchmark | 0.5828 [0.518, 0.647] | 44.00% | 57.67% | 57.14% | 61.33% | 57.57% | 9.9 ms | 4.26 MB |
| **DistilHuBERT (SSL)** | **MLAAD Benchmark** | 0.4417 [0.377, 0.505] | 55.33% | 43.67% | 45.45% | 66.67% | 42.92% | 42.5 ms | 95.6 MB |
| LCNN-MFM | MLAAD Benchmark | 0.5698 [0.504, 0.635] | 45.33% | 55.67% | 55.03% | 62.00% | 55.51% | 6.7 ms | 2.03 MB |
| SE-ResNet-18 | MLAAD Benchmark | 0.4114 [0.347, 0.473] | 57.33% | 40.00% | 40.00% | 40.00% | 40.00% | 9.9 ms | 4.26 MB |

---

## 📈 Publication Figures Included

| Figure | Description | File Path |
| :--- | :--- | :--- |
| **Figure 1** | RQ1 Cross-Lingual Generalization across Indian Language Families | [`results/plots/fig1_rq1_cross_lingual.png`](results/plots/fig1_rq1_cross_lingual.png) |
| **Figure 2** | RQ2 Codec Robustness Curves under Progressive Compression | [`results/plots/fig2_rq2_codec_robustness.png`](results/plots/fig2_rq2_codec_robustness.png) |
| **Figure 3** | RQ3 Pareto Frontier (CPU Latency vs. AUROC) | [`results/plots/fig3_rq3_latency_vs_accuracy.png`](results/plots/fig3_rq3_latency_vs_accuracy.png) |
| **Figure 4** | Dual Held-Out Evaluation (AUROC & Balanced Accuracy) | [`results/plots/fig4_dual_held_out_comparison.png`](results/plots/fig4_dual_held_out_comparison.png) |
| **Figure 5** | Precision-Recall Frontier under Default vs. Tuned Thresholds | [`results/plots/fig5_precision_recall_frontier.png`](results/plots/fig5_precision_recall_frontier.png) |
| **Figure 6** | Head-to-Head: DistilHuBERT vs. Compact Baselines | [`results/plots/fig6_distilhubert_vs_baselines_cross_lingual.png`](results/plots/fig6_distilhubert_vs_baselines_cross_lingual.png) |
| **Figure 7** | External Benchmark Generalization (MLAAD vs. ASVspoof 5) | [`results/plots/fig7_external_benchmark_generalization.png`](results/plots/fig7_external_benchmark_generalization.png) |
| **Figure 8** | Complete Codec Invariance Trajectory Across All Architectures | [`results/plots/fig8_codec_robustness_all_models.png`](results/plots/fig8_codec_robustness_all_models.png) |

---

## 📁 Repository Structure

```
.
├── app/
│   └── demo_app.py                # Interactive Streamlit forensic demonstration app
├── checkpoints/                   # Trained model weights (DistilHuBERT, SE-ResNet, LCNN, SVM)
│   ├── cnn_augmented_best.pt      # Codec-augmented 2D-CNN (1.29 MB)
│   ├── cnn_clean_best.pt          # Clean-trained 2D-CNN (1.29 MB)
│   ├── distilhubert_best.pt       # SOTA DistilHuBERT FP16 model (95.6 MB)
│   ├── lcnn_mfm_augmented_best.pt # LCNN with Max-Feature-Map (2.03 MB)
│   ├── model_a_svm_rbf.joblib     # Calibrated LFCC RBF-SVM (1.58 MB)
│   └── se_resnet_augmented_best.pt# Squeeze-and-Excitation ResNet-18 (4.26 MB)
├── configs/                       # Experiment hyperparameter configs (YAML)
├── data/
│   ├── benchmarks/                # External OOD manifests (ASVspoof 5, MLAAD)
│   ├── manifests/                 # Master JSONL manifests and audit statistics
│   └── splits/                    # Prompt-disjoint train, val, and held-out test CSV splits
├── results/
│   ├── plots/                     # High-resolution publication plots (Figures 1-8)
│   └── tables/                    # CSV evaluation matrices with bootstrap CIs
├── scripts/                       # Dataset download, audit, and benchmark evaluation scripts
├── src/
│   ├── evaluation/                # AUROC, EER, accuracy, and bootstrap confidence interval metrics
│   ├── explainability/            # Time-frequency occlusion attribution
│   ├── features/                  # Log-Mel, LFCC, and statistical feature extractors
│   ├── models/                    # DistilHuBERT, LCNN-MFM, SE-ResNet, SVM, and Ensemble
│   ├── preprocessing/             # Audio standardizer, resampler, and split generator
│   ├── robustness/                # FFmpeg MP3/Opus simulation and online augmentation
│   └── training/                  # FP16 AMP PyTorch training loops with cosine annealing
├── tests/                         # Pytest test suite (21 unit tests covering all modules)
├── MAJOR_PROJECT_REPORT.md        # Comprehensive academic project report & paper draft
├── requirements.txt
└── README.md
```

---

## ⚡ Quickstart & Reproduction Guide

### 1. Environment Setup
```powershell
# Clone the repository
git clone https://github.com/AbhishekYadav410/Cross-Lingual-Audio-Deepfake-Detection.git
cd Cross-Lingual-Audio-Deepfake-Detection

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies (PyTorch with CUDA 12.4 recommended for GPU training)
pip install -r requirements.txt
```

### 2. Run Test Suite
```powershell
$env:PYTHONPATH = "."
pytest tests/ -v
```

### 3. Launch Interactive Forensic Demonstration App
Experience live waveform analysis, log-mel spectrogram visualization, multi-model probability comparison, and lossy codec simulation:
```powershell
streamlit run app/demo_app.py
```

### 4. Reproduce Model Training
```powershell
# Train DistilHuBERT with FP16 Mixed Precision on GPU:
python -m src.training.train_distilhubert

# Train SE-ResNet-18 with on-the-fly Codec Augmentation:
python -m src.training.train_advanced --model se_resnet --use_codec_aug

# Train LCNN-MFM with on-the-fly Codec Augmentation:
python -m src.training.train_advanced --model lcnn --use_codec_aug

# Train Calibrated LFCC RBF-SVM Baseline:
python -m src.training.train_classical
```

---

## 📖 Citation

If you use this benchmark, code, or findings in your academic research, please cite:

```bibtex
@article{yadav2026indicdeepfake,
  title   = {Cross-Lingual Generalization and Codec Robustness of AI-Generated Speech Detectors for Indian Languages},
  author  = {Yadav, Abhishek},
  journal = {Major Project Research Report, Department of Computer Science and Engineering},
  year    = {2026}
}
```

---

## 📄 License
This repository is released under the **MIT License**.

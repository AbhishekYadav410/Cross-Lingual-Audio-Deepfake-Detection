import sys
import os
sys.path.insert(0, ".")

from app.demo_app import load_models
import numpy as np

def main():
    print("Testing demo_app load_models()...")
    extractor, classical_extractor, codec_sim, cnn_clean, lcnn_model, se_resnet, svm_model, ensemble = load_models()
    print("Models loaded successfully!")
    dummy_audio = np.zeros(48000, dtype=np.float32)
    spec = extractor.extract_log_mel(dummy_audio)
    feat_1d = classical_extractor.extract_vector(dummy_audio).reshape(1, -1)
    print(f"Spec shape: {spec.shape}, Classical feat shape: {feat_1d.shape}")
    print("All demo_app components verified!")

if __name__ == "__main__":
    main()

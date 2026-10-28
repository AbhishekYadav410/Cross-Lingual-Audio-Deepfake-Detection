import torch

print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Capability: {torch.cuda.get_device_capability(0)}")
    print(f"Total Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**2):.1f} MB")
else:
    print("CUDA is NOT enabled in current PyTorch build.")

try:
    import transformers
    print(f"Transformers Version: {transformers.__version__}")
except ImportError:
    print("Transformers is NOT installed.")

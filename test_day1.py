import torch
import torchvision.models as models
print("✅ PyTorch version:", torch.__version__)
print("✅ CUDA available:", torch.cuda.is_available())

# Test DenseNet-121
model = models.densenet121(pretrained=True)
print("✅ DenseNet-121 loaded successfully!")
print("✅ Model parameters:", sum(p.numel() for p in model.parameters()))

# Test basic inference
dummy_input = torch.randn(1, 3, 224, 224)
with torch.no_grad():
    output = model(dummy_input)
print(f"✅ Inference test: Input {dummy_input.shape} -> Output {output.shape}")
print("\n🎉 Day 1 Foundation Complete!")
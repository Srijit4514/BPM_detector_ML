import torch
import torch.nn as nn
from src.model import PhysNetED
import onnx
import onnxruntime as ort
import numpy as np

# Configuration
MODEL_PATH = "models/physnet_single_debug.pth"
ONNX_PATH = "models/physnet.onnx"
WINDOW_SIZE = 64

def export_to_onnx():
    """Converts the PhysNet model to ONNX format."""
    model = PhysNetED()
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location='cpu'))
        print(f"Loaded model from {MODEL_PATH}")
    except:
        print(f"Model not found at {MODEL_PATH}. Exporting uninitialized model.")

    model.eval()

    # Create dummy input: (B, C, T, H, W)
    dummy_input = torch.randn(1, 3, WINDOW_SIZE, 64, 64)

    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        ONNX_PATH,
        export_params=True,
        opset_version=18,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output']
    )
    print(f"Successfully exported model to {ONNX_PATH}")

def verify_onnx():
    """Verify the exported ONNX model with a dummy input."""
    # Load ONNX model
    onnx_model = onnx.load(ONNX_PATH)
    onnx.checker.check_model(onnx_model)
    print("ONNX model verification passed!")

    # Run inference with ONNX Runtime
    ort_session = ort.InferenceSession(ONNX_PATH)

    # Input data
    dummy_input = np.random.randn(1, 3, WINDOW_SIZE, 64, 64).astype(np.float32)

    # Run session
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input}
    ort_outs = ort_session.run(None, ort_inputs)

    print(f"Inference output shape: {ort_outs[0].shape}")

if __name__ == "__main__":
    export_to_onnx()
    verify_onnx()

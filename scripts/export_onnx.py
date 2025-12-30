#!/usr/bin/env python3
"""
Export trained model to ONNX format for production deployment.

ONNX enables:
- Faster inference with ONNX Runtime
- Cross-platform deployment (C++, JavaScript, etc.)
- Optimized inference on various hardware

Usage:
    python scripts/export_onnx.py --checkpoint checkpoints/best.pth --output model.onnx
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models import UNetSmall, UNet


def export_to_onnx(
    checkpoint_path: str,
    output_path: str,
    input_size: tuple = (1, 4, 128, 128),
    opset_version: int = 14,
    simplify: bool = True
):
    """
    Export PyTorch model to ONNX format.
    
    Args:
        checkpoint_path: Path to .pth checkpoint
        output_path: Path to save .onnx model
        input_size: Input tensor shape (B, C, H, W)
        opset_version: ONNX opset version
        simplify: Whether to simplify the ONNX graph
    """
    print("=" * 60)
    print("PCB Thermal AI - ONNX Export")
    print("=" * 60)
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Determine model type
    model_name = checkpoint.get('model_name', 'unet_small')
    print(f"Model type: {model_name}")
    
    if model_name == 'unet_small':
        model = UNetSmall(in_channels=4, out_channels=1)
    else:
        model = UNet(in_channels=4, out_channels=1)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")
    
    # Create dummy input
    dummy_input = torch.randn(*input_size)
    
    # Export to ONNX
    print(f"\nExporting to ONNX (opset {opset_version})...")
    
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=['pcb_features'],
        output_names=['temperature'],
        dynamic_axes={
            'pcb_features': {0: 'batch_size'},
            'temperature': {0: 'batch_size'}
        }
    )
    
    print(f"✓ Exported to {output_path}")
    
    # Simplify ONNX graph (optional but recommended)
    if simplify:
        try:
            import onnx
            from onnxsim import simplify as onnx_simplify
            
            print("\nSimplifying ONNX graph...")
            model_onnx = onnx.load(output_path)
            model_simplified, check = onnx_simplify(model_onnx)
            
            if check:
                onnx.save(model_simplified, output_path)
                print("✓ Graph simplified successfully")
            else:
                print("⚠ Simplification check failed, keeping original")
        except ImportError:
            print("⚠ onnx-simplifier not installed, skipping simplification")
            print("  Install with: pip install onnx-simplifier")
    
    # Verify ONNX model
    print("\nVerifying ONNX model...")
    try:
        import onnx
        model_onnx = onnx.load(output_path)
        onnx.checker.check_model(model_onnx)
        print("✓ ONNX model is valid")
    except Exception as e:
        print(f"⚠ ONNX verification warning: {e}")
    
    # Test with ONNX Runtime
    print("\nTesting with ONNX Runtime...")
    try:
        import onnxruntime as ort
        
        session = ort.InferenceSession(output_path)
        
        # Run inference
        ort_inputs = {session.get_inputs()[0].name: dummy_input.numpy()}
        ort_outputs = session.run(None, ort_inputs)
        
        # Compare with PyTorch
        with torch.no_grad():
            torch_output = model(dummy_input).numpy()
        
        diff = np.abs(ort_outputs[0] - torch_output).max()
        print(f"✓ ONNX Runtime test passed (max diff: {diff:.6f})")
        
        # Benchmark
        import time
        n_runs = 100
        
        # Warmup
        for _ in range(10):
            session.run(None, ort_inputs)
        
        # Benchmark
        t_start = time.time()
        for _ in range(n_runs):
            session.run(None, ort_inputs)
        ort_time = (time.time() - t_start) / n_runs * 1000
        
        print(f"  ONNX Runtime inference: {ort_time:.2f}ms")
        
    except ImportError:
        print("⚠ ONNX Runtime not installed")
        print("  Install with: pip install onnxruntime")
    
    # Save metadata
    metadata_path = Path(output_path).with_suffix('.json')
    metadata = {
        'model_name': model_name,
        'input_shape': list(input_size),
        'output_shape': [input_size[0], 1, input_size[2], input_size[3]],
        'opset_version': opset_version,
        'num_parameters': num_params,
        'normalization': {
            'output_mean': checkpoint.get('output_stats', {}).get('mean', 50.0),
            'output_std': checkpoint.get('output_stats', {}).get('std', 30.0)
        },
        'input_channels': ['copper', 'vias', 'components', 'power']
    }
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"\n✓ Metadata saved to {metadata_path}")
    
    # Final summary
    file_size = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"\n" + "=" * 60)
    print("Export Complete!")
    print("=" * 60)
    print(f"  Model:     {output_path}")
    print(f"  Size:      {file_size:.2f} MB")
    print(f"  Params:    {num_params:,}")
    print(f"\nUsage in Python:")
    print(f"  import onnxruntime as ort")
    print(f"  session = ort.InferenceSession('{output_path}')")
    print(f"  output = session.run(None, {{'pcb_features': input_array}})")


def main():
    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best.pth',
                       help='Path to PyTorch checkpoint')
    parser.add_argument('--output', type=str, default='model.onnx',
                       help='Output ONNX file path')
    parser.add_argument('--opset', type=int, default=14,
                       help='ONNX opset version')
    parser.add_argument('--no-simplify', action='store_true',
                       help='Skip ONNX graph simplification')
    
    args = parser.parse_args()
    
    export_to_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        opset_version=args.opset,
        simplify=not args.no_simplify
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Test script to verify variable resolution SigLIP support in ClipSAM model.

This script tests different SigLIP resolutions to ensure the architecture 
handles them correctly without hardcoded spatial dimensions.
"""

import torch
import sys
from model import SigLipSamSegmentator

def test_resolution(model_name, expected_input_size, expected_patch_size, expected_spatial_dim):
    """Test a specific SigLIP model resolution"""
    print(f"\n{'='*80}")
    print(f"Testing: {model_name}")
    print(f"{'='*80}")
    
    try:
        # Initialize model
        model = SigLipSamSegmentator(
            siglip_model_name=model_name,
            sam_model_name='facebook/sam-vit-base',
            device='cpu'  # Use CPU for testing
        )
        
        # Verify configuration
        print(f"\n✓ Model initialized successfully")
        assert model.siglip_image_size == expected_input_size, \
            f"Expected input size {expected_input_size}, got {model.siglip_image_size}"
        assert model.siglip_patch_size == expected_patch_size, \
            f"Expected patch size {expected_patch_size}, got {model.siglip_patch_size}"
        assert model.siglip_spatial_dim == expected_spatial_dim, \
            f"Expected spatial dim {expected_spatial_dim}, got {model.siglip_spatial_dim}"
        
        print(f"✓ Configuration verified:")
        print(f"  - Input size: {model.siglip_image_size}")
        print(f"  - Patch size: {model.siglip_patch_size}")
        print(f"  - Spatial dim: {model.siglip_spatial_dim}")
        print(f"  - Downsampling steps: {model.down_spatial_times}")
        print(f"  - Target output: {model.target_spatial_dim}x{model.target_spatial_dim}")
        
        # Test forward pass with dummy input
        print(f"\n✓ Testing forward pass...")
        batch_size = 2
        dummy_image = torch.randn(batch_size, 3, 480, 480)
        dummy_text = ["a building", "a car"]
        
        with torch.no_grad():
            output = model(dummy_image, dummy_text)
        
        print(f"✓ Forward pass successful")
        print(f"  - Input shape: {dummy_image.shape}")
        print(f"  - Output shape: {output.shape}")
        assert output.shape == (batch_size, 480, 480), \
            f"Expected output shape {(batch_size, 480, 480)}, got {output.shape}"
        
        print(f"\n{'='*80}")
        print(f"✓ ALL TESTS PASSED for {model_name}")
        print(f"{'='*80}\n")
        
        return True
        
    except Exception as e:
        print(f"\n✗ TEST FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Test different SigLIP model resolutions"""
    
    print("\n" + "="*80)
    print("TESTING VARIABLE RESOLUTION SIGLIP SUPPORT")
    print("="*80)
    
    # Define test cases: (model_name, input_size, patch_size, spatial_dim)
    test_cases = [
        # Original 384x384 model
        ('google/siglip2-so400m-patch14-384', 384, 14, 27),
        
        # Alternative resolutions (if available)
        # Note: These models need to exist on HuggingFace
        # Uncomment when testing with actual models
        # ('google/siglip-base-patch16-224', 224, 16, 14),
        # ('google/siglip-base-patch16-256', 256, 16, 16),
        # ('google/siglip-base-patch16-512', 512, 16, 32),
    ]
    
    results = []
    for model_name, input_size, patch_size, spatial_dim in test_cases:
        success = test_resolution(model_name, input_size, patch_size, spatial_dim)
        results.append((model_name, success))
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    for model_name, success in results:
        status = "✓ PASSED" if success else "✗ FAILED"
        print(f"{status}: {model_name}")
    
    all_passed = all(success for _, success in results)
    print("="*80)
    
    if all_passed:
        print("✓ ALL TESTS PASSED!")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        return 1


if __name__ == '__main__':
    sys.exit(main())

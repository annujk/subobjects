"""
Demo script for Safe DirectSAM Training
Shows how to use the enhanced training system to prevent NaN explosions.
"""

import os
import torch
import logging
from safe_trainer import SafeDirectSAMTrainer


def demo_safe_training():
    """Demonstrate safe training setup and configuration."""
    print("🛡️  DirectSAM Safe Training Demo")
    print("=" * 50)
    
    # Setup logging to see safety messages
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Create safe trainer with ultra-conservative settings
    print("\n1. Creating ultra-safe trainer configuration...")
    trainer = SafeDirectSAMTrainer(
        config_phase="ultra_safe",  # Most conservative settings
        enable_recovery=True,       # Enable recovery from checkpoints
        enable_monitoring=True,     # Enable gradient monitoring
        output_dir="runs/demo-safe-training"
    )
    
    # Show configuration details
    config = trainer.config
    print(f"   ✓ Learning rate: {config.learning_rate:.2e} (very conservative)")
    print(f"   ✓ Gradient clipping: {config.max_grad_norm} (strong)")
    print(f"   ✓ Mixed precision: {config.enable_fp16} (disabled for stability)")
    print(f"   ✓ Save frequency: every {config.save_steps} steps")
    print(f"   ✓ Warmup ratio: {config.warmup_ratio} (extended)")
    
    # Show safety features
    print("\n2. Safety features enabled:")
    print("   ✓ NaN detection with automatic halt")
    print("   ✓ Gradient explosion monitoring")
    print("   ✓ Loss spike detection")
    print("   ✓ Adaptive learning rate reduction")
    print("   ✓ Emergency checkpoint saving")
    print("   ✓ Model health validation")
    print("   ✓ Automatic recovery from safe checkpoints")
    
    # Show available configurations
    print("\n3. Available training phases:")
    phases = [
        ("ultra_safe", "5e-7", "0.3", "Extended warmup, no mixed precision"),
        ("safe", "1e-6", "0.5", "Conservative settings, stable training"),
        ("moderate", "2e-6", "0.7", "Moderate settings after stability proven"),
        ("normal", "5e-6", "1.0", "Standard settings with mixed precision")
    ]
    
    for phase, lr, clip, desc in phases:
        print(f"   • {phase:12} - LR: {lr:6}, Clip: {clip}, {desc}")
    
    print("\n4. Progressive training approach:")
    print("   1. Start with 'ultra_safe' phase for stability")
    print("   2. Progress to 'safe' phase once stable")
    print("   3. Move to 'moderate' phase for better performance")
    print("   4. Finally use 'normal' phase for optimal training")
    
    print("\n5. Recovery capabilities:")
    if trainer.recovery_system:
        print("   ✓ Automatic detection of previous NaN explosions")
        print("   ✓ Safe checkpoint identification and loading")
        print("   ✓ Model health validation before resuming")
        print("   ✓ Ultra-conservative recovery settings")
        print("   ✓ Emergency backup creation")
    
    print("\n6. To start training:")
    print("   # Basic safe training")
    print("   python safe_trainer.py --config-phase safe --epochs 3")
    print("")
    print("   # Progressive training (recommended)")
    print("   python safe_trainer.py --progressive --epochs 1")
    print("")
    print("   # Ultra-safe recovery training")
    print("   python safe_trainer.py --config-phase ultra_safe --epochs 1")
    
    print("\n7. Monitoring and alerts:")
    print("   • Real-time gradient norm tracking")
    print("   • Automatic learning rate reduction on instability")
    print("   • Loss spike detection and warnings")
    print("   • Visual gradient health reports saved as PNG")
    print("   • Comprehensive logging of all safety events")
    
    print("\n🎯 Expected benefits:")
    print("   ✓ No more NaN loss explosions")
    print("   ✓ Stable, recoverable training")
    print("   ✓ Automatic intervention on gradient issues")
    print("   ✓ Production-ready bacterial segmentation model")
    print("   ✓ Detailed monitoring and debugging information")


def check_environment():
    """Check if the environment is ready for safe training."""
    print("\n🔍 Environment Check")
    print("=" * 30)
    
    # Check PyTorch
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"✓ CUDA devices: {torch.cuda.device_count()}")
    
    # Check required modules
    try:
        from transformers import AutoModelForSemanticSegmentation
        print("✓ Transformers library available")
    except ImportError:
        print("✗ Transformers library missing")
    
    try:
        from datasets import load_dataset
        print("✓ Datasets library available")
    except ImportError:
        print("✗ Datasets library missing")
    
    # Check if original trainer exists
    if os.path.exists("trainer.py"):
        print("✓ Original trainer.py found")
    else:
        print("✗ Original trainer.py not found")
    
    # Check safety modules
    safety_modules = [
        "enhanced_training_config.py",
        "gradient_health_monitor.py", 
        "training_callbacks.py",
        "safe_training_recovery.py",
        "safe_trainer.py"
    ]
    
    print("\nSafety modules:")
    for module in safety_modules:
        if os.path.exists(module):
            print(f"✓ {module}")
        else:
            print(f"✗ {module}")


if __name__ == "__main__":
    check_environment()
    demo_safe_training()
    
    print("\n" + "=" * 70)
    print("🚀 Ready to start safe DirectSAM training!")
    print("   Run: python safe_trainer.py --help for all options")
    print("=" * 70)
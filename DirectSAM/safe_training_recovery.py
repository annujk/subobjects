"""
Safe Training Recovery Script for DirectSAM
Handles recovery from NaN explosions and provides safe restart mechanisms.
"""

import os
import json
import torch
import shutil
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from transformers import AutoModelForSemanticSegmentation, AutoImageProcessor
from enhanced_training_config import RECOVERY_CONFIG, ULTRA_SAFE_CONFIG


class SafeTrainingRecovery:
    """Handles recovery from training failures and NaN explosions."""
    
    def __init__(self, training_dir: str, backup_dir: Optional[str] = None):
        self.training_dir = Path(training_dir)
        self.backup_dir = Path(backup_dir) if backup_dir else self.training_dir / "backups"
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger("SafeRecovery")
        
        # Ensure backup directory exists
        self.backup_dir.mkdir(parents=True, exist_ok=True)
    
    def detect_nan_explosion(self) -> Optional[Dict]:
        """Detect if a NaN explosion occurred by checking recovery info."""
        recovery_file = self.training_dir / "nan_recovery_info.json"
        
        if recovery_file.exists():
            with open(recovery_file, 'r') as f:
                recovery_info = json.load(f)
            
            self.logger.warning(f"NaN explosion detected at step {recovery_info['nan_detected_at_step']}")
            return recovery_info
        
        return None
    
    def find_safe_checkpoint(self, max_steps_back: int = 500) -> Optional[str]:
        """Find the most recent safe checkpoint before NaN explosion."""
        recovery_info = self.detect_nan_explosion()
        
        if recovery_info and recovery_info.get('last_good_checkpoint'):
            checkpoint_path = self.training_dir / recovery_info['last_good_checkpoint']
            if checkpoint_path.exists():
                self.logger.info(f"Found safe checkpoint: {checkpoint_path}")
                return str(checkpoint_path)
        
        # Fallback: find most recent checkpoint
        checkpoints = self._find_all_checkpoints()
        
        if recovery_info:
            nan_step = recovery_info['nan_detected_at_step']
            # Find checkpoint before NaN explosion
            safe_checkpoints = [
                (path, step) for path, step in checkpoints 
                if step < nan_step - 50  # 50 step safety margin
            ]
            if safe_checkpoints:
                return max(safe_checkpoints, key=lambda x: x[1])[0]
        
        elif checkpoints:
            return max(checkpoints, key=lambda x: x[1])[0]
        
        self.logger.warning("No safe checkpoint found")
        return None
    
    def _find_all_checkpoints(self) -> List[Tuple[str, int]]:
        """Find all available checkpoints with their step numbers."""
        checkpoints = []
        
        if self.training_dir.exists():
            for item in self.training_dir.iterdir():
                if item.is_dir() and item.name.startswith('checkpoint-'):
                    try:
                        step = int(item.name.split('-')[1])
                        checkpoints.append((str(item), step))
                    except (IndexError, ValueError):
                        continue
        
        return sorted(checkpoints, key=lambda x: x[1])
    
    def backup_current_state(self) -> str:
        """Backup current training state before recovery attempt."""
        timestamp = torch.utils.data.get_worker_info()
        if timestamp is None:
            import time
            timestamp = int(time.time())
        else:
            timestamp = timestamp.id
            
        backup_path = self.backup_dir / f"backup_{timestamp}"
        
        if self.training_dir.exists():
            shutil.copytree(self.training_dir, backup_path, dirs_exist_ok=True)
            self.logger.info(f"Current state backed up to {backup_path}")
            return str(backup_path)
        
        return ""
    
    def load_safe_model(self, checkpoint_path: str) -> Tuple[any, any]:
        """Load model and image processor from safe checkpoint."""
        try:
            self.logger.info(f"Loading model from safe checkpoint: {checkpoint_path}")
            
            # Load model
            model = AutoModelForSemanticSegmentation.from_pretrained(
                checkpoint_path,
                num_labels=1,
                ignore_mismatched_sizes=True
            )
            
            # Try to load image processor from checkpoint, fallback to original
            try:
                image_processor = AutoImageProcessor.from_pretrained(checkpoint_path)
            except:
                # Fallback to original checkpoint
                original_checkpoint = "chendelong/DirectSAM-1800px-0424"
                image_processor = AutoImageProcessor.from_pretrained(
                    original_checkpoint, 
                    reduce_labels=True
                )
                self.logger.warning(f"Using original image processor from {original_checkpoint}")
            
            self.logger.info("Model and processor loaded successfully")
            return model, image_processor
            
        except Exception as e:
            self.logger.error(f"Failed to load model from checkpoint: {e}")
            raise
    
    def create_recovery_training_args(self, checkpoint_path: str, phase: str = "ultra_safe"):
        """Create ultra-conservative training arguments for recovery."""
        config = ULTRA_SAFE_CONFIG if phase == "ultra_safe" else RECOVERY_CONFIG
        
        # Create recovery output directory
        recovery_dir = self.training_dir.parent / f"recovery_{phase}"
        recovery_dir.mkdir(exist_ok=True)
        
        training_args = config.get_training_args()
        training_args.output_dir = str(recovery_dir)
        
        # Override with even more conservative settings for recovery
        training_args.learning_rate = 1e-7  # Ultra conservative
        training_args.max_grad_norm = 0.2   # Very strong clipping
        training_args.save_steps = 50       # More frequent saves
        training_args.logging_steps = 1     # Log every step
        training_args.warmup_ratio = 0.5    # Extended warmup
        
        self.logger.info(f"Created recovery training args with LR={training_args.learning_rate:.2e}")
        return training_args
    
    def validate_model_health(self, model) -> Dict[str, any]:
        """Validate that the loaded model is healthy."""
        health_report = {
            "has_nan_weights": False,
            "has_inf_weights": False,
            "weight_stats": {},
            "is_healthy": True
        }
        
        for name, param in model.named_parameters():
            if torch.isnan(param).any():
                health_report["has_nan_weights"] = True
                health_report["is_healthy"] = False
                self.logger.error(f"NaN weights found in {name}")
            
            if torch.isinf(param).any():
                health_report["has_inf_weights"] = True
                health_report["is_healthy"] = False
                self.logger.error(f"Inf weights found in {name}")
        
        # Basic statistics
        all_params = torch.cat([p.flatten() for p in model.parameters()])
        health_report["weight_stats"] = {
            "mean": float(all_params.mean()),
            "std": float(all_params.std()),
            "min": float(all_params.min()),
            "max": float(all_params.max())
        }
        
        if health_report["is_healthy"]:
            self.logger.info("Model health check passed")
        else:
            self.logger.error("Model health check failed")
        
        return health_report
    
    def create_recovery_script(self, output_path: str = "recover_training.py"):
        """Create a standalone recovery script."""
        script_content = '''"""
Automated Training Recovery Script
Generated by SafeTrainingRecovery
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from safe_training_recovery import SafeTrainingRecovery
from enhanced_training_config import RECOVERY_CONFIG
from training_callbacks import get_safety_callbacks
from transformers import Trainer
import torch.distributed as dist

def main():
    # Initialize distributed training if needed
    if torch.cuda.device_count() > 1:
        dist.init_process_group(backend='nccl')
    
    # Setup recovery
    recovery = SafeTrainingRecovery("runs/finetune-directsam-ade20k-5ep-512px")
    
    # Find and load safe checkpoint
    safe_checkpoint = recovery.find_safe_checkpoint()
    if not safe_checkpoint:
        print("No safe checkpoint found. Cannot recover.")
        return
    
    # Backup current state
    recovery.backup_current_state()
    
    # Load model
    model, image_processor = recovery.load_safe_model(safe_checkpoint)
    
    # Validate model health
    health = recovery.validate_model_health(model)
    if not health["is_healthy"]:
        print("Model is not healthy. Manual intervention required.")
        return
    
    # Setup recovery training
    training_args = recovery.create_recovery_training_args(safe_checkpoint)
    
    # Load dataset (reuse from original trainer)
    from trainer import dataset, transforms
    dataset.set_transform(transforms)
    
    # Create trainer with safety callbacks
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        callbacks=get_safety_callbacks()
    )
    
    print("Starting recovery training with ultra-safe settings...")
    trainer.train(resume_from_checkpoint=safe_checkpoint)

if __name__ == "__main__":
    main()
'''
        
        with open(output_path, 'w') as f:
            f.write(script_content)
        
        # Make executable
        os.chmod(output_path, 0o755)
        self.logger.info(f"Recovery script created: {output_path}")
    
    def get_recovery_recommendations(self) -> List[str]:
        """Get specific recommendations for recovery."""
        recommendations = []
        
        recovery_info = self.detect_nan_explosion()
        safe_checkpoint = self.find_safe_checkpoint()
        
        if recovery_info:
            recommendations.extend([
                f"NaN explosion detected at step {recovery_info['nan_detected_at_step']}",
                f"Last good step was {recovery_info['last_good_step']} with loss {recovery_info['last_good_loss']:.6f}",
                ""
            ])
        
        if safe_checkpoint:
            recommendations.extend([
                f"✓ Safe checkpoint found: {safe_checkpoint}",
                "✓ Can proceed with recovery",
                ""
            ])
        else:
            recommendations.extend([
                "✗ No safe checkpoint found",
                "✗ May need to restart from pretrained model",
                ""
            ])
        
        recommendations.extend([
            "Recovery Actions:",
            "1. Backup current training state",
            "2. Load model from safe checkpoint",
            "3. Use ultra-conservative training settings:",
            "   - Learning rate: 1e-7 (100x reduction)",
            "   - Gradient clipping: 0.2 (very strong)",
            "   - No mixed precision",
            "   - Extended warmup: 50%",
            "   - Frequent checkpoints: every 50 steps",
            "4. Enable all safety callbacks",
            "5. Monitor closely for first 500 steps",
            "6. Gradually increase learning rate if stable"
        ])
        
        return recommendations


def emergency_recovery_cli():
    """Command-line interface for emergency recovery."""
    import argparse
    
    parser = argparse.ArgumentParser(description="DirectSAM Training Recovery Tool")
    parser.add_argument("--training-dir", default="runs/finetune-directsam-ade20k-5ep-512px",
                       help="Training directory to recover")
    parser.add_argument("--backup-dir", help="Directory for backups")
    parser.add_argument("--create-script", action="store_true",
                       help="Create recovery script")
    parser.add_argument("--check-only", action="store_true",
                       help="Only check for issues, don't recover")
    
    args = parser.parse_args()
    
    # Initialize recovery
    recovery = SafeTrainingRecovery(args.training_dir, args.backup_dir)
    
    # Print recommendations
    print("\n=== DirectSAM Training Recovery Analysis ===")
    recommendations = recovery.get_recovery_recommendations()
    for rec in recommendations:
        print(rec)
    
    if args.create_script:
        recovery.create_recovery_script()
        print("\nRecovery script created: recover_training.py")
    
    if not args.check_only:
        print("\nTo proceed with recovery, run: python recover_training.py")


if __name__ == "__main__":
    emergency_recovery_cli()
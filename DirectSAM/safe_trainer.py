"""
Safe DirectSAM Trainer with NaN Explosion Prevention
Enhanced version of the original trainer with comprehensive safety measures.
"""

import cv2
import numpy as np
from PIL import Image as PILImage
import os
import logging
import torch
import torchvision.transforms as transforms
import torch.distributed as dist

from datasets import Dataset, load_dataset
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation, Trainer

# Import our safety modules
from enhanced_training_config import SafeTrainingConfig, SAFE_CONFIG, ULTRA_SAFE_CONFIG
from training_callbacks import get_safety_callbacks
from gradient_health_monitor import GradientHealthMonitor
from safe_training_recovery import SafeTrainingRecovery


def annotation_to_label(label_map, line_thickness=3):
    """
    Convert annotation to boundary label (from original trainer).
    """
    label_map = np.array(label_map)
    all_contours = []
    for label_idx in np.unique(label_map):
        mask = (label_map == label_idx).astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        all_contours.append(contours)
    h, w = label_map.shape
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    for contours in all_contours:
        cv2.drawContours(canvas, contours, -1, (1, 1, 1), line_thickness)
    label = PILImage.fromarray(canvas[:, :, 0], mode='L')
    return label


def transforms_with_safety(example_batch, image_processor):
    """Transform function with additional safety checks."""
    images = [x.convert("RGB") for x in example_batch["image"]]
    labels = [annotation_to_label(x) for x in example_batch["annotation"]]
    
    # Add input validation
    valid_images = []
    valid_labels = []
    
    for img, lbl in zip(images, labels):
        if img.size[0] > 0 and img.size[1] > 0:  # Valid dimensions
            valid_images.append(img)
            valid_labels.append(lbl)
    
    if not valid_images:
        # Return dummy batch if all images are invalid
        dummy_img = PILImage.new('RGB', (224, 224), color='black')
        dummy_lbl = PILImage.new('L', (224, 224), color=0)
        valid_images = [dummy_img]
        valid_labels = [dummy_lbl]
    
    inputs = image_processor(valid_images, valid_labels, do_reduce_labels=False)
    return inputs


class SafeDirectSAMTrainer:
    """Enhanced DirectSAM trainer with comprehensive safety measures."""
    
    def __init__(self, 
                 config_phase="safe",  # "ultra_safe", "safe", "moderate"
                 enable_recovery=True,
                 enable_monitoring=True,
                 output_dir="runs/safe-directsam-finetuning"):
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("SafeDirectSAM")
        
        # Configuration
        self.config_phase = config_phase
        self.enable_recovery = enable_recovery
        self.enable_monitoring = enable_monitoring
        self.output_dir = output_dir
        
        # Initialize components
        self.config = self._get_config()
        self.model = None
        self.image_processor = None
        self.dataset = None
        self.trainer = None
        
        # Safety components
        self.recovery_system = None
        if enable_recovery:
            self.recovery_system = SafeTrainingRecovery(output_dir)
    
    def _get_config(self):
        """Get training configuration based on phase."""
        configs = {
            "ultra_safe": ULTRA_SAFE_CONFIG,
            "safe": SAFE_CONFIG,
            "moderate": SafeTrainingConfig().get_progressive_config("moderate"),
            "normal": SafeTrainingConfig().get_progressive_config("normal")
        }
        
        config = configs.get(self.config_phase, SAFE_CONFIG)
        config.output_dir = self.output_dir
        return config
    
    def setup_model_and_processor(self, checkpoint="chendelong/DirectSAM-1800px-0424", 
                                 input_resolution=512):
        """Setup model and image processor with safety checks."""
        self.logger.info(f"Setting up model from checkpoint: {checkpoint}")
        
        # Check for recovery first
        if self.enable_recovery and self.recovery_system:
            recovery_checkpoint = self.recovery_system.find_safe_checkpoint()
            if recovery_checkpoint:
                self.logger.info(f"Recovery checkpoint found, using: {recovery_checkpoint}")
                self.model, self.image_processor = self.recovery_system.load_safe_model(recovery_checkpoint)
                
                # Validate model health
                health = self.recovery_system.validate_model_health(self.model)
                if not health["is_healthy"]:
                    self.logger.error("Recovered model is not healthy, falling back to original")
                    self.model = None
                    self.image_processor = None
        
        # Load from original checkpoint if no recovery model
        if self.model is None:
            self.model = AutoModelForSemanticSegmentation.from_pretrained(
                checkpoint, 
                num_labels=1, 
                ignore_mismatched_sizes=True
            )
            self.image_processor = AutoImageProcessor.from_pretrained(
                checkpoint, 
                reduce_labels=True
            )
        
        # Configure input resolution
        self.image_processor.size['height'] = input_resolution
        self.image_processor.size['width'] = input_resolution
        
        # Model info
        if torch.distributed.get_rank() == 0:
            total_params = self.model.num_parameters() / 1e6
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad) / 1e6
            self.logger.info(f"Model loaded: {total_params:.1f}M total, {trainable_params:.1f}M trainable parameters")
    
    def setup_dataset(self, dataset_name="scene_parse_150", split="train"):
        """Setup dataset with safety transformations."""
        self.logger.info(f"Loading dataset: {dataset_name}")
        
        self.dataset = load_dataset(dataset_name, split=split)
        
        # Create safe transform function
        def safe_transforms(example_batch):
            return transforms_with_safety(example_batch, self.image_processor)
        
        self.dataset.set_transform(safe_transforms)
        
        if torch.distributed.get_rank() == 0:
            self.logger.info(f"Dataset loaded: {len(self.dataset)} samples")
    
    def create_trainer(self, num_train_epochs=3, per_device_batch_size=4):
        """Create trainer with safety callbacks and monitoring."""
        training_args = self.config.get_training_args(
            num_train_epochs=num_train_epochs,
            per_device_batch_size=per_device_batch_size
        )
        
        # Get safety callbacks
        callbacks = get_safety_callbacks(self.config)
        
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.dataset,
            callbacks=callbacks
        )
        
        self.logger.info(f"Trainer created with {len(callbacks)} safety callbacks")
        self.logger.info(f"Training configuration: LR={training_args.learning_rate:.2e}, "
                        f"grad_clip={training_args.max_grad_norm}, fp16={training_args.fp16}")
    
    def train_with_safety(self, resume_from_checkpoint=None):
        """Start training with comprehensive safety monitoring."""
        self.logger.info("Starting safe DirectSAM training...")
        
        # Pre-training safety check
        if self.enable_recovery and self.recovery_system:
            nan_info = self.recovery_system.detect_nan_explosion()
            if nan_info:
                self.logger.warning("Previous NaN explosion detected. Recommendations:")
                for rec in self.recovery_system.get_recovery_recommendations():
                    self.logger.warning(f"  {rec}")
        
        # Create backup before training
        if self.enable_recovery and self.recovery_system:
            backup_path = self.recovery_system.backup_current_state()
            self.logger.info(f"Training state backed up to: {backup_path}")
        
        try:
            # Start training
            self.trainer.train(resume_from_checkpoint=resume_from_checkpoint)
            self.logger.info("Training completed successfully!")
            
        except Exception as e:
            self.logger.error(f"Training failed with error: {e}")
            
            # Emergency save
            if self.trainer:
                emergency_path = os.path.join(self.output_dir, "emergency_save")
                self.trainer.save_model(emergency_path)
                self.logger.info(f"Emergency model save completed: {emergency_path}")
            
            # Generate recovery information
            if self.enable_recovery and self.recovery_system:
                recovery_file = os.path.join(self.output_dir, "training_failure_info.json")
                with open(recovery_file, 'w') as f:
                    import json
                    failure_info = {
                        "error": str(e),
                        "error_type": type(e).__name__,
                        "config_phase": self.config_phase,
                        "recommendations": [
                            "Check gradient_health_report.png for gradient analysis",
                            "Use recovery script: python safe_training_recovery.py",
                            "Consider using ultra_safe configuration",
                            "Reduce learning rate further if needed"
                        ]
                    }
                    json.dump(failure_info, f, indent=2)
                
                self.logger.info(f"Training failure info saved to: {recovery_file}")
            
            raise
    
    def progressive_training(self, phases=["ultra_safe", "safe", "moderate"], 
                           epochs_per_phase=1):
        """Progressive training with increasing aggressiveness."""
        for i, phase in enumerate(phases):
            self.logger.info(f"Starting progressive training phase {i+1}/{len(phases)}: {phase}")
            
            # Update configuration
            self.config = self.config.get_progressive_config(phase)
            
            # Recreate trainer with new config
            self.create_trainer(num_train_epochs=epochs_per_phase)
            
            # Resume from previous phase or start fresh
            resume_checkpoint = None
            if i > 0:
                # Find most recent checkpoint
                checkpoints = [d for d in os.listdir(self.output_dir) 
                              if d.startswith('checkpoint-')]
                if checkpoints:
                    checkpoint_nums = [int(c.split('-')[1]) for c in checkpoints 
                                     if c.split('-')[1].isdigit()]
                    if checkpoint_nums:
                        latest_checkpoint = max(checkpoint_nums)
                        resume_checkpoint = f"checkpoint-{latest_checkpoint}"
            
            # Train this phase
            self.train_with_safety(resume_from_checkpoint=resume_checkpoint)
            
            self.logger.info(f"Phase {phase} completed successfully")


def main():
    """Main training function with safety measures."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Safe DirectSAM Training")
    parser.add_argument("--config-phase", default="safe", 
                       choices=["ultra_safe", "safe", "moderate", "normal"],
                       help="Training safety configuration")
    parser.add_argument("--output-dir", default="runs/safe-directsam-finetuning",
                       help="Output directory for training")
    parser.add_argument("--resume-from", help="Resume from checkpoint")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per device")
    parser.add_argument("--progressive", action="store_true",
                       help="Use progressive training phases")
    parser.add_argument("--input-resolution", type=int, default=512,
                       help="Input image resolution")
    
    args = parser.parse_args()
    
    # Initialize distributed training
    dist.init_process_group(backend='nccl')
    
    # Create safe trainer
    trainer = SafeDirectSAMTrainer(
        config_phase=args.config_phase,
        output_dir=args.output_dir
    )
    
    # Setup model and dataset
    trainer.setup_model_and_processor(input_resolution=args.input_resolution)
    trainer.setup_dataset()
    trainer.create_trainer(
        num_train_epochs=args.epochs,
        per_device_batch_size=args.batch_size
    )
    
    # Start training
    if args.progressive:
        trainer.progressive_training()
    else:
        trainer.train_with_safety(resume_from_checkpoint=args.resume_from)


if __name__ == '__main__':
    main()
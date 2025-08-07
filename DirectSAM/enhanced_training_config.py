"""
Enhanced Training Configuration for DirectSAM Fine-tuning
Addresses NaN loss explosion with conservative, stable settings.
"""

import torch
from transformers import TrainingArguments


class SafeTrainingConfig:
    """Conservative training configuration to prevent NaN explosions."""
    
    def __init__(self, 
                 output_dir="runs/safe-directsam-finetuning",
                 learning_rate=1e-6,  # Very conservative
                 max_grad_norm=0.5,   # Strong gradient clipping
                 enable_fp16=False,   # Disable mixed precision initially
                 save_steps=100,      # Frequent checkpoints
                 warmup_ratio=0.2):   # Longer warmup
        
        self.learning_rate = learning_rate
        self.max_grad_norm = max_grad_norm
        self.enable_fp16 = enable_fp16
        self.save_steps = save_steps
        self.warmup_ratio = warmup_ratio
        self.output_dir = output_dir
    
    def get_training_args(self, num_train_epochs=3, per_device_batch_size=4):
        """Get safe training arguments."""
        return TrainingArguments(
            output_dir=self.output_dir,
            learning_rate=self.learning_rate,
            num_train_epochs=num_train_epochs,
            per_device_train_batch_size=per_device_batch_size,
            gradient_accumulation_steps=1,
            max_grad_norm=self.max_grad_norm,  # Critical: gradient clipping
            
            # Checkpointing and saving
            save_total_limit=10,  # Keep more checkpoints for recovery
            save_steps=self.save_steps,
            save_strategy="steps",  # Save by steps, not epochs
            
            # Logging and monitoring
            logging_steps=1,  # Log every step for monitoring
            logging_strategy="steps",
            
            # Performance and stability
            dataloader_num_workers=4,
            dataloader_prefetch_factor=4,
            remove_unused_columns=False,
            
            # Precision and optimization
            fp16=self.enable_fp16,  # Start with fp32 for stability
            bf16=False,  # Avoid mixed precision initially
            
            # Learning rate scheduling
            warmup_ratio=self.warmup_ratio,
            lr_scheduler_type="cosine_with_restarts",
            
            # Evaluation and validation
            do_eval=False,  # Focus on training stability first
            
            # Hub and reporting
            push_to_hub=False,
            report_to=None,  # Disable wandb/tensorboard initially
            
            # Safety measures
            ignore_data_skip=True,  # Important for recovery
            load_best_model_at_end=False,  # Avoid loading during training
        )
    
    def get_progressive_config(self, phase="ultra_safe"):
        """Get progressive training configurations for different phases."""
        configs = {
            "ultra_safe": {
                "learning_rate": 5e-7,
                "max_grad_norm": 0.3,
                "enable_fp16": False,
                "warmup_ratio": 0.3
            },
            "safe": {
                "learning_rate": 1e-6,
                "max_grad_norm": 0.5,
                "enable_fp16": False,
                "warmup_ratio": 0.2
            },
            "moderate": {
                "learning_rate": 2e-6,
                "max_grad_norm": 0.7,
                "enable_fp16": False,
                "warmup_ratio": 0.15
            },
            "normal": {
                "learning_rate": 5e-6,
                "max_grad_norm": 1.0,
                "enable_fp16": True,
                "warmup_ratio": 0.1
            }
        }
        
        config = configs.get(phase, configs["safe"])
        self.learning_rate = config["learning_rate"]
        self.max_grad_norm = config["max_grad_norm"]
        self.enable_fp16 = config["enable_fp16"]
        self.warmup_ratio = config["warmup_ratio"]
        
        return self


# Pre-defined safe configurations
ULTRA_SAFE_CONFIG = SafeTrainingConfig(
    learning_rate=5e-7,
    max_grad_norm=0.3,
    enable_fp16=False,
    warmup_ratio=0.3
)

SAFE_CONFIG = SafeTrainingConfig(
    learning_rate=1e-6,
    max_grad_norm=0.5,
    enable_fp16=False,
    warmup_ratio=0.2
)

RECOVERY_CONFIG = SafeTrainingConfig(
    learning_rate=2e-7,  # Ultra conservative for recovery
    max_grad_norm=0.2,
    enable_fp16=False,
    warmup_ratio=0.5,    # Extended warmup
    save_steps=50        # More frequent saves during recovery
)
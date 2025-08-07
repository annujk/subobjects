"""
Training Callbacks for DirectSAM Fine-tuning
Provides NaN detection, gradient monitoring, and safety interventions.
"""

import torch
import numpy as np
import os
import json
import logging
from typing import Dict, Any, Optional
from transformers import TrainerCallback, TrainerState, TrainerControl, TrainingArguments
from gradient_health_monitor import GradientHealthMonitor


class NaNDetectionCallback(TrainerCallback):
    """Callback to detect NaN losses and halt training safely."""
    
    def __init__(self, save_on_nan=True, create_recovery_info=True):
        self.save_on_nan = save_on_nan
        self.create_recovery_info = create_recovery_info
        self.nan_detected = False
        self.last_good_step = 0
        self.last_good_loss = float('inf')
        
        # Setup logging
        self.logger = logging.getLogger("NaNDetection")
        self.logger.setLevel(logging.INFO)
    
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Check for NaN in loss after each logging step."""
        if state.log_history:
            latest_log = state.log_history[-1]
            
            # Check training loss
            if 'train_loss' in latest_log:
                loss = latest_log['train_loss']
                
                if np.isnan(loss) or np.isinf(loss):
                    self.logger.error(f"NaN/Inf loss detected at step {state.global_step}: {loss}")
                    self.nan_detected = True
                    
                    if self.save_on_nan:
                        self._handle_nan_detection(args, state, control, **kwargs)
                    
                    # Stop training
                    control.should_training_stop = True
                    return control
                
                else:
                    # Update last good state
                    self.last_good_step = state.global_step
                    self.last_good_loss = loss
        
        return control
    
    def _handle_nan_detection(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Handle NaN detection by saving recovery information."""
        recovery_info = {
            "nan_detected_at_step": state.global_step,
            "last_good_step": self.last_good_step,
            "last_good_loss": self.last_good_loss,
            "last_good_checkpoint": None,
            "recommended_actions": [
                "Load checkpoint from before NaN explosion",
                "Reduce learning rate by 10x",
                "Enable stronger gradient clipping",
                "Consider disabling mixed precision",
                "Increase checkpoint frequency"
            ],
            "training_args": {
                "learning_rate": args.learning_rate,
                "max_grad_norm": getattr(args, 'max_grad_norm', None),
                "fp16": args.fp16,
                "save_steps": args.save_steps
            }
        }
        
        # Find most recent checkpoint
        if os.path.exists(args.output_dir):
            checkpoints = [d for d in os.listdir(args.output_dir) if d.startswith('checkpoint-')]
            if checkpoints:
                # Sort by checkpoint number
                checkpoint_nums = [int(c.split('-')[1]) for c in checkpoints if c.split('-')[1].isdigit()]
                if checkpoint_nums:
                    latest_checkpoint = max(checkpoint_nums)
                    if latest_checkpoint <= self.last_good_step:
                        recovery_info["last_good_checkpoint"] = f"checkpoint-{latest_checkpoint}"
        
        # Save recovery information
        recovery_path = os.path.join(args.output_dir, "nan_recovery_info.json")
        with open(recovery_path, 'w') as f:
            json.dump(recovery_info, f, indent=2)
        
        self.logger.info(f"Recovery information saved to {recovery_path}")


class GradientMonitoringCallback(TrainerCallback):
    """Callback to monitor gradient health during training."""
    
    def __init__(self, 
                 explosion_threshold=10.0,
                 intervention_enabled=True,
                 auto_save_on_warning=True):
        
        self.monitor = GradientHealthMonitor(explosion_threshold=explosion_threshold)
        self.intervention_enabled = intervention_enabled
        self.auto_save_on_warning = auto_save_on_warning
        self.intervention_count = 0
        
        # Setup logging
        self.logger = logging.getLogger("GradientMonitoring")
        self.logger.setLevel(logging.INFO)
    
    def on_step_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Monitor gradients after each training step."""
        model = kwargs.get('model')
        if model is None:
            return control
        
        # Update gradient monitoring
        health_report = self.monitor.update(model, state.global_step)
        
        # Log gradient health
        if state.global_step % args.logging_steps == 0:
            self.logger.info(
                f"Step {state.global_step}: Grad norm={health_report['grad_norm']:.6f}, "
                f"Status={health_report['health_status']}, "
                f"Risk={health_report['explosion_risk']:.2f}"
            )
        
        # Handle warnings and interventions
        if health_report['warning'] and self.intervention_enabled:
            self._handle_gradient_warning(args, state, control, health_report, **kwargs)
        
        # Auto-save on warnings
        if health_report['health_status'] in ['WARNING', 'CRITICAL'] and self.auto_save_on_warning:
            self._emergency_save(args, state, **kwargs)
        
        # Stop on critical gradient explosion
        if health_report['health_status'] == 'CRITICAL':
            self.logger.critical("CRITICAL gradient explosion detected - stopping training")
            control.should_training_stop = True
        
        return control
    
    def _handle_gradient_warning(self, args: TrainingArguments, state: TrainerState, 
                                control: TrainerControl, health_report: Dict[str, Any], **kwargs):
        """Handle gradient warnings with automatic interventions."""
        self.intervention_count += 1
        
        self.logger.warning(f"Gradient intervention #{self.intervention_count}: {health_report['recommendation']}")
        
        # Get optimizer for learning rate adjustment
        optimizer = kwargs.get('optimizer')
        if optimizer and health_report['explosion_risk'] > 0.7:
            # Reduce learning rate temporarily
            for param_group in optimizer.param_groups:
                old_lr = param_group['lr']
                param_group['lr'] *= 0.5  # Halve the learning rate
                self.logger.warning(f"Reduced learning rate from {old_lr:.2e} to {param_group['lr']:.2e}")
    
    def _emergency_save(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Perform emergency checkpoint save."""
        trainer = kwargs.get('trainer')
        if trainer:
            emergency_path = os.path.join(args.output_dir, f"emergency-checkpoint-{state.global_step}")
            trainer.save_model(emergency_path)
            self.logger.info(f"Emergency checkpoint saved to {emergency_path}")
    
    def on_train_end(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Save gradient health report at the end of training."""
        report_path = os.path.join(args.output_dir, "gradient_health_report.png")
        self.monitor.save_report(report_path)


class LossSpikeDetectionCallback(TrainerCallback):
    """Callback to detect sudden loss spikes that may precede NaN explosions."""
    
    def __init__(self, spike_threshold=5.0, window_size=10):
        self.spike_threshold = spike_threshold
        self.window_size = window_size
        self.loss_history = []
        self.spike_count = 0
        
        # Setup logging
        self.logger = logging.getLogger("LossSpike")
        self.logger.setLevel(logging.INFO)
    
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Monitor loss for sudden spikes."""
        if state.log_history:
            latest_log = state.log_history[-1]
            
            if 'train_loss' in latest_log:
                current_loss = latest_log['train_loss']
                
                if not (np.isnan(current_loss) or np.isinf(current_loss)):
                    self.loss_history.append(current_loss)
                    
                    # Keep only recent history
                    if len(self.loss_history) > self.window_size:
                        self.loss_history.pop(0)
                    
                    # Check for spikes
                    if len(self.loss_history) >= 3:
                        recent_mean = np.mean(self.loss_history[:-1])  # Exclude current loss
                        
                        if current_loss > recent_mean * self.spike_threshold:
                            self.spike_count += 1
                            self.logger.warning(
                                f"Loss spike detected at step {state.global_step}: "
                                f"{current_loss:.6f} vs recent mean {recent_mean:.6f} "
                                f"(spike #{self.spike_count})"
                            )
                            
                            # Multiple spikes indicate instability
                            if self.spike_count >= 3:
                                self.logger.error("Multiple loss spikes detected - training may be unstable")
        
        return control


class AdaptiveLearningRateCallback(TrainerCallback):
    """Callback to adaptively reduce learning rate on training instability."""
    
    def __init__(self, 
                 patience=5,
                 reduction_factor=0.5,
                 min_lr=1e-8,
                 loss_threshold=1.5):
        
        self.patience = patience
        self.reduction_factor = reduction_factor
        self.min_lr = min_lr
        self.loss_threshold = loss_threshold
        
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.reductions = 0
        
        # Setup logging
        self.logger = logging.getLogger("AdaptiveLR")
        self.logger.setLevel(logging.INFO)
    
    def on_log(self, args: TrainingArguments, state: TrainerState, control: TrainerControl, **kwargs):
        """Monitor loss and adjust learning rate if needed."""
        if state.log_history:
            latest_log = state.log_history[-1]
            
            if 'train_loss' in latest_log:
                current_loss = latest_log['train_loss']
                
                if not (np.isnan(current_loss) or np.isinf(current_loss)):
                    
                    # Check if loss improved
                    if current_loss < self.best_loss:
                        self.best_loss = current_loss
                        self.patience_counter = 0
                    elif current_loss > self.best_loss * self.loss_threshold:
                        self.patience_counter += 1
                        
                        if self.patience_counter >= self.patience:
                            self._reduce_learning_rate(args, state, **kwargs)
                            self.patience_counter = 0
        
        return control
    
    def _reduce_learning_rate(self, args: TrainingArguments, state: TrainerState, **kwargs):
        """Reduce learning rate when loss plateaus or increases."""
        optimizer = kwargs.get('optimizer')
        if optimizer:
            for param_group in optimizer.param_groups:
                old_lr = param_group['lr']
                new_lr = max(old_lr * self.reduction_factor, self.min_lr)
                
                if new_lr != old_lr:
                    param_group['lr'] = new_lr
                    self.reductions += 1
                    
                    self.logger.info(
                        f"Learning rate reduced at step {state.global_step}: "
                        f"{old_lr:.2e} -> {new_lr:.2e} (reduction #{self.reductions})"
                    )


def get_safety_callbacks(config=None):
    """Get a comprehensive set of safety callbacks."""
    callbacks = [
        NaNDetectionCallback(save_on_nan=True),
        GradientMonitoringCallback(
            explosion_threshold=getattr(config, 'explosion_threshold', 10.0),
            intervention_enabled=True
        ),
        LossSpikeDetectionCallback(spike_threshold=3.0),
        AdaptiveLearningRateCallback(patience=3)
    ]
    
    return callbacks
"""
Gradient Health Monitor for DirectSAM Training
Detects gradient explosions and provides early warning systems.
"""

import torch
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
import matplotlib.pyplot as plt
from collections import deque
import warnings


class GradientHealthMonitor:
    """Monitor gradient health and detect potential explosions."""
    
    def __init__(self, 
                 explosion_threshold=10.0,
                 spike_threshold=3.0,
                 history_size=100,
                 warning_threshold=2.0):
        
        self.explosion_threshold = explosion_threshold
        self.spike_threshold = spike_threshold
        self.warning_threshold = warning_threshold
        self.history_size = history_size
        
        # Gradient tracking
        self.grad_norms = deque(maxlen=history_size)
        self.grad_means = deque(maxlen=history_size)
        self.grad_stds = deque(maxlen=history_size)
        self.step_count = 0
        
        # Health indicators
        self.health_status = "HEALTHY"
        self.last_warning_step = -1
        self.explosion_detected = False
        
        # Statistics
        self.baseline_norm = None
        self.baseline_std = None
        
        # Setup logging
        self.logger = logging.getLogger("GradientHealth")
        self.logger.setLevel(logging.INFO)
    
    def compute_gradient_norm(self, model) -> float:
        """Compute the L2 norm of all gradients."""
        total_norm = 0.0
        param_count = 0
        
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_count += 1
        
        if param_count == 0:
            return 0.0
        
        return (total_norm ** 0.5)
    
    def compute_gradient_stats(self, model) -> Dict[str, float]:
        """Compute comprehensive gradient statistics."""
        grad_values = []
        
        for p in model.parameters():
            if p.grad is not None:
                grad_values.extend(p.grad.data.flatten().cpu().numpy())
        
        if not grad_values:
            return {"mean": 0.0, "std": 0.0, "max": 0.0, "min": 0.0}
        
        grad_array = np.array(grad_values)
        return {
            "mean": float(np.mean(grad_array)),
            "std": float(np.std(grad_array)),
            "max": float(np.max(grad_array)),
            "min": float(np.min(grad_array)),
            "median": float(np.median(grad_array))
        }
    
    def update(self, model, step: int) -> Dict[str, any]:
        """Update gradient monitoring and return health status."""
        self.step_count = step
        
        # Compute gradient metrics
        grad_norm = self.compute_gradient_norm(model)
        grad_stats = self.compute_gradient_stats(model)
        
        # Store history
        self.grad_norms.append(grad_norm)
        self.grad_means.append(grad_stats["mean"])
        self.grad_stds.append(grad_stats["std"])
        
        # Establish baseline if not set
        if self.baseline_norm is None and len(self.grad_norms) >= 10:
            self.baseline_norm = np.mean(list(self.grad_norms)[:10])
            self.baseline_std = np.std(list(self.grad_norms)[:10])
            self.logger.info(f"Baseline gradient norm established: {self.baseline_norm:.6f} ± {self.baseline_std:.6f}")
        
        # Analyze health
        health_report = self._analyze_health(grad_norm, grad_stats)
        
        return {
            "grad_norm": grad_norm,
            "grad_stats": grad_stats,
            "health_status": self.health_status,
            "warning": health_report.get("warning", False),
            "recommendation": health_report.get("recommendation", ""),
            "explosion_risk": health_report.get("explosion_risk", 0.0)
        }
    
    def _analyze_health(self, grad_norm: float, grad_stats: Dict[str, float]) -> Dict[str, any]:
        """Analyze gradient health and detect issues."""
        warning = False
        recommendation = ""
        explosion_risk = 0.0
        
        # Check for NaN or inf
        if np.isnan(grad_norm) or np.isinf(grad_norm):
            self.health_status = "CRITICAL"
            self.explosion_detected = True
            return {
                "warning": True,
                "recommendation": "STOP TRAINING: NaN/Inf detected in gradients!",
                "explosion_risk": 1.0
            }
        
        # Check for gradient explosion
        if grad_norm > self.explosion_threshold:
            self.health_status = "CRITICAL"
            explosion_risk = 1.0
            warning = True
            recommendation = f"GRADIENT EXPLOSION: norm={grad_norm:.6f} > threshold={self.explosion_threshold}"
            self.explosion_detected = True
        
        # Check for gradient spikes (compared to recent history)
        elif len(self.grad_norms) >= 5:
            recent_mean = np.mean(list(self.grad_norms)[-5:])
            if grad_norm > recent_mean * self.spike_threshold:
                self.health_status = "WARNING"
                explosion_risk = 0.7
                warning = True
                recommendation = f"Gradient spike detected: {grad_norm:.6f} vs recent mean {recent_mean:.6f}"
        
        # Check for growing instability
        elif len(self.grad_norms) >= 10:
            recent_trend = self._compute_trend()
            if recent_trend > self.warning_threshold:
                self.health_status = "CAUTION"
                explosion_risk = 0.5
                warning = True
                recommendation = f"Increasing gradient trend detected: {recent_trend:.3f}"
        
        # All clear
        else:
            self.health_status = "HEALTHY"
            explosion_risk = 0.0
        
        # Log warnings
        if warning and step - self.last_warning_step > 10:  # Avoid spam
            self.logger.warning(f"Step {self.step_count}: {recommendation}")
            self.last_warning_step = step
        
        return {
            "warning": warning,
            "recommendation": recommendation,
            "explosion_risk": explosion_risk
        }
    
    def _compute_trend(self) -> float:
        """Compute the trend in gradient norms over recent history."""
        if len(self.grad_norms) < 10:
            return 0.0
        
        recent_norms = list(self.grad_norms)[-10:]
        x = np.arange(len(recent_norms))
        slope = np.polyfit(x, recent_norms, 1)[0]
        
        # Normalize by current gradient norm to get relative trend
        current_norm = recent_norms[-1]
        if current_norm > 0:
            return slope / current_norm
        return 0.0
    
    def get_recommendations(self) -> List[str]:
        """Get training recommendations based on current health."""
        recommendations = []
        
        if self.health_status == "CRITICAL":
            recommendations.extend([
                "STOP TRAINING immediately",
                "Load previous checkpoint",
                "Reduce learning rate by 10x",
                "Enable gradient clipping with max_norm=0.1",
                "Consider disabling mixed precision"
            ])
        
        elif self.health_status == "WARNING":
            recommendations.extend([
                "Reduce learning rate by 2-5x",
                "Strengthen gradient clipping",
                "Save checkpoint immediately",
                "Monitor next 50 steps closely"
            ])
        
        elif self.health_status == "CAUTION":
            recommendations.extend([
                "Consider reducing learning rate",
                "Enable more frequent checkpointing",
                "Monitor gradient trends"
            ])
        
        return recommendations
    
    def save_report(self, filepath: str = "gradient_health_report.png"):
        """Save a visual report of gradient health."""
        if len(self.grad_norms) < 2:
            self.logger.warning("Insufficient data for report generation")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # Gradient norms over time
        axes[0, 0].plot(list(self.grad_norms))
        axes[0, 0].set_title("Gradient Norms Over Time")
        axes[0, 0].set_ylabel("Gradient Norm")
        axes[0, 0].grid(True)
        
        # Add threshold lines
        if self.baseline_norm:
            axes[0, 0].axhline(y=self.baseline_norm, color='g', linestyle='--', label='Baseline')
        axes[0, 0].axhline(y=self.explosion_threshold, color='r', linestyle='--', label='Explosion Threshold')
        axes[0, 0].legend()
        
        # Gradient statistics
        axes[0, 1].plot(list(self.grad_means), label='Mean')
        axes[0, 1].plot(list(self.grad_stds), label='Std')
        axes[0, 1].set_title("Gradient Statistics")
        axes[0, 1].set_ylabel("Value")
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Health status distribution
        if len(self.grad_norms) > 0:
            axes[1, 0].hist(list(self.grad_norms), bins=20, alpha=0.7)
            axes[1, 0].set_title("Gradient Norm Distribution")
            axes[1, 0].set_xlabel("Gradient Norm")
            axes[1, 0].set_ylabel("Frequency")
        
        # Status summary
        axes[1, 1].text(0.1, 0.8, f"Current Status: {self.health_status}", fontsize=12, weight='bold')
        axes[1, 1].text(0.1, 0.6, f"Steps Monitored: {len(self.grad_norms)}", fontsize=10)
        axes[1, 1].text(0.1, 0.4, f"Explosions Detected: {self.explosion_detected}", fontsize=10)
        if self.baseline_norm:
            axes[1, 1].text(0.1, 0.2, f"Baseline Norm: {self.baseline_norm:.6f}", fontsize=10)
        axes[1, 1].set_xlim(0, 1)
        axes[1, 1].set_ylim(0, 1)
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        self.logger.info(f"Gradient health report saved to {filepath}")
    
    def is_safe_to_continue(self) -> bool:
        """Check if it's safe to continue training."""
        return self.health_status in ["HEALTHY", "CAUTION"] and not self.explosion_detected


class GradientClipCallback:
    """Callback for safe gradient clipping with monitoring."""
    
    def __init__(self, max_norm=1.0, monitor: Optional[GradientHealthMonitor] = None):
        self.max_norm = max_norm
        self.monitor = monitor
        self.clipping_applied = 0
        self.total_steps = 0
    
    def __call__(self, model, step: int) -> Dict[str, float]:
        """Apply gradient clipping and monitor health."""
        self.total_steps += 1
        
        # Compute gradient norm before clipping
        grad_norm_before = self.monitor.compute_gradient_norm(model) if self.monitor else 0.0
        
        # Apply clipping
        grad_norm_after = torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_norm)
        
        # Track clipping events
        if grad_norm_after >= self.max_norm * 0.99:  # Nearly clipped
            self.clipping_applied += 1
        
        return {
            "grad_norm_before": grad_norm_before,
            "grad_norm_after": float(grad_norm_after),
            "clipping_applied": grad_norm_after >= self.max_norm * 0.99,
            "clipping_ratio": self.clipping_applied / self.total_steps if self.total_steps > 0 else 0.0
        }
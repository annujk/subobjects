"""
Minimal Safety Patch for Original DirectSAM Trainer
Adds basic NaN detection and gradient clipping without major changes.
"""

import cv2
import numpy as np
from PIL import Image as PILImage
import torch
import torchvision.transforms as transforms
import torch.distributed as dist
from datasets import Dataset, load_dataset
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation, TrainingArguments, Trainer
from transformers import TrainerCallback, TrainerState, TrainerControl


class MinimalSafetyCallback(TrainerCallback):
    """Minimal safety callback for NaN detection and basic gradient clipping."""
    
    def __init__(self, max_grad_norm=1.0):
        self.max_grad_norm = max_grad_norm
        self.nan_detected = False
        
    def on_step_end(self, args, state, control, **kwargs):
        # Apply gradient clipping
        model = kwargs.get('model')
        if model:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.max_grad_norm)
        return control
    
    def on_log(self, args, state, control, **kwargs):
        # Check for NaN in loss
        if state.log_history:
            latest_log = state.log_history[-1]
            if 'train_loss' in latest_log:
                loss = latest_log['train_loss']
                if np.isnan(loss) or np.isinf(loss):
                    print(f"🚨 NaN/Inf loss detected at step {state.global_step}: {loss}")
                    print("Training halted. Use safe_trainer.py for recovery.")
                    control.should_training_stop = True
                    self.nan_detected = True
        return control


def annotation_to_label(label_map, line_thickness=3):
    """Original annotation conversion function."""
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


def transforms(example_batch):
    """Original transform function."""
    images = [x.convert("RGB") for x in example_batch["image"]]
    labels = [annotation_to_label(x) for x in example_batch["annotation"]]
    inputs = image_processor(images, labels, do_reduce_labels=False)
    return inputs


if __name__=='__main__':
    
    dist.init_process_group(backend='nccl')

    dataset = load_dataset("scene_parse_150", split="train")
    dataset.set_transform(transforms)

    checkpoint = "chendelong/DirectSAM-1800px-0424"
    model = AutoModelForSemanticSegmentation.from_pretrained(checkpoint, num_labels=1, ignore_mismatched_sizes=True)
    image_processor = AutoImageProcessor.from_pretrained(checkpoint, reduce_labels=True)

    input_resolution = 512
    image_processor.size['height'] = input_resolution
    image_processor.size['width'] = input_resolution
    
    if torch.distributed.get_rank() == 0:
        print(model)
        print(f"Number of parameters: {model.num_parameters()/1e6}M,  trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)/1e6}M")
        print(dataset)

    # 🛡️ SAFETY ENHANCEMENT: More conservative settings
    training_args = TrainingArguments(
        output_dir=f'runs/safer-directsam-ade20k-5ep-512px',
        learning_rate=2e-6,  # Reduced from 5e-5 (2.5x reduction)
        num_train_epochs=3,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=1,
        max_grad_norm=0.5,   # 🛡️ NEW: Gradient clipping
        save_total_limit=5,   # Keep more checkpoints
        dataloader_num_workers=4,
        dataloader_prefetch_factor=4,
        save_strategy="steps", # 🛡️ NEW: Save by steps
        save_steps=200,       # 🛡️ NEW: More frequent saves
        do_eval=False,
        logging_steps=1,
        remove_unused_columns=False,
        push_to_hub=False,
        fp16=False           # 🛡️ NEW: Disable mixed precision for stability
    )

    # 🛡️ SAFETY ENHANCEMENT: Add safety callback
    safety_callback = MinimalSafetyCallback(max_grad_norm=0.5)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        callbacks=[safety_callback]  # 🛡️ NEW: Safety callback
    )

    print("🛡️ Starting training with basic safety measures...")
    print("   - Reduced learning rate (2e-6)")
    print("   - Gradient clipping (0.5)")
    print("   - Disabled mixed precision")
    print("   - More frequent checkpoints")
    print("   - NaN detection enabled")
    print("")
    print("For comprehensive safety features, use: python safe_trainer.py")

    trainer.train()
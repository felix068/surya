#!/usr/bin/env python3
"""
Simple fine-tuning script for Surya text detection model.
Uses YOLO format dataset (images + txt labels with bboxes).
Only requires line-level annotations (no character or word level needed).

Dataset structure:
    data/
    ├── images/
    │   ├── img1.jpg
    │   ├── img2.jpg
    │   └── ...
    └── labels/
        ├── img1.txt  # YOLO format: class x_center y_center width height
        ├── img2.txt
        └── ...

YOLO label format (normalized 0-1):
    0 0.5 0.3 0.8 0.05  # class=0, center_x, center_y, width, height
    0 0.5 0.4 0.8 0.05
"""

import os
import cv2
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F

# Import Surya model
from surya.detection.model.config import EfficientViTConfig
from surya.detection.model.encoderdecoder import EfficientViTForSemanticSegmentation


# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Dataset paths
    IMAGES_DIR = "data/images"
    LABELS_DIR = "data/labels"

    # Model paths
    PRETRAINED_MODEL = "s3://text_detection/2025_05_07"
    OUTPUT_DIR = "./finetuned_model"

    # Training hyperparameters
    EPOCHS = 20
    BATCH_SIZE = 4
    LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 1e-4

    # Image settings (same as Surya)
    IMAGE_SIZE = 512

    # ImageNet normalization (same as Surya)
    MEAN = np.array([0.485, 0.456, 0.406])
    STD = np.array([0.229, 0.224, 0.225])

    # Training settings
    NUM_WORKERS = 4
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    SAVE_EVERY = 5  # Save checkpoint every N epochs

    # Loss weights
    TEXT_LOSS_WEIGHT = 1.0
    AFFINITY_LOSS_WEIGHT = 0.0  # Set to 0 since we don't have affinity labels


# ============================================================================
# DATASET
# ============================================================================

class YOLOTextDataset(Dataset):
    """
    Dataset for text line detection from YOLO format annotations.
    Converts bounding boxes to segmentation masks.
    """

    def __init__(self, images_dir, labels_dir, image_size=512):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.image_size = image_size

        # Get all image files
        self.image_files = sorted([
            f for f in os.listdir(images_dir)
            if f.endswith(('.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'))
        ])

        print(f"Found {len(self.image_files)} images in {images_dir}")

    def __len__(self):
        return len(self.image_files)

    def load_yolo_labels(self, label_path, orig_width, orig_height):
        """
        Load YOLO format labels and convert to pixel coordinates.

        YOLO format: class x_center y_center width height (all normalized 0-1)
        Returns: List of bboxes in pixel coordinates [[x_min, y_min, x_max, y_max], ...]
        """
        bboxes = []

        if not os.path.exists(label_path):
            return bboxes

        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                # Parse YOLO format (normalized)
                class_id = int(parts[0])
                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

                # Convert to pixel coordinates
                x_center_px = x_center * orig_width
                y_center_px = y_center * orig_height
                width_px = width * orig_width
                height_px = height * orig_height

                # Convert to corner format
                x_min = x_center_px - width_px / 2
                y_min = y_center_px - height_px / 2
                x_max = x_center_px + width_px / 2
                y_max = y_center_px + height_px / 2

                bboxes.append([x_min, y_min, x_max, y_max])

        return bboxes

    def create_mask_from_bboxes(self, bboxes, orig_size, target_size):
        """
        Create a binary segmentation mask from bounding boxes.

        Args:
            bboxes: List of [x_min, y_min, x_max, y_max] in original image coordinates
            orig_size: (width, height) of original image
            target_size: (width, height) of target mask (512x512)

        Returns:
            mask: Binary mask of shape (target_size, target_size)
        """
        # Create mask at original size
        orig_width, orig_height = orig_size
        mask_orig = np.zeros((orig_height, orig_width), dtype=np.uint8)

        # Draw each bbox as filled rectangle
        for bbox in bboxes:
            x_min, y_min, x_max, y_max = bbox
            x_min = int(np.clip(x_min, 0, orig_width))
            y_min = int(np.clip(y_min, 0, orig_height))
            x_max = int(np.clip(x_max, 0, orig_width))
            y_max = int(np.clip(y_max, 0, orig_height))

            cv2.rectangle(mask_orig, (x_min, y_min), (x_max, y_max),
                         color=255, thickness=-1)  # -1 = filled

        # Resize mask to target size
        mask_resized = cv2.resize(mask_orig, (target_size, target_size),
                                 interpolation=cv2.INTER_NEAREST)

        # Normalize to [0, 1]
        mask_resized = mask_resized.astype(np.float32) / 255.0

        return mask_resized

    def preprocess_image(self, img):
        """
        Preprocess image exactly like Surya's inference.
        Same as simple_inf.py preprocessing.
        """
        # Resize to 512x512 (double resize for accuracy)
        img.thumbnail((self.image_size, self.image_size), Image.Resampling.LANCZOS)
        img = img.resize((self.image_size, self.image_size), Image.Resampling.LANCZOS)

        # Convert to numpy array [H, W, C]
        img_np = np.asarray(img, dtype=np.float32)

        # Rescale from [0, 255] to [0, 1]
        img_np = img_np / 255.0

        # Normalize with ImageNet stats
        img_np = (img_np - Config.MEAN) / Config.STD

        # Convert to [C, H, W] format
        img_np = np.transpose(img_np, (2, 0, 1))

        return img_np

    def __getitem__(self, idx):
        # Load image
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        orig_width, orig_height = image.size

        # Load YOLO labels
        label_name = os.path.splitext(img_name)[0] + '.txt'
        label_path = os.path.join(self.labels_dir, label_name)
        bboxes = self.load_yolo_labels(label_path, orig_width, orig_height)

        # Create mask from bboxes
        text_mask = self.create_mask_from_bboxes(
            bboxes, (orig_width, orig_height), self.image_size
        )

        # Preprocess image
        image_tensor = self.preprocess_image(image)

        # Convert to tensors
        image_tensor = torch.from_numpy(image_tensor).float()
        text_mask = torch.from_numpy(text_mask).float()

        # Create dummy affinity mask (zeros, since we don't have affinity labels)
        affinity_mask = torch.zeros_like(text_mask)

        return image_tensor, text_mask, affinity_mask, img_name


# ============================================================================
# TRAINING
# ============================================================================

class DiceLoss(nn.Module):
    """
    Dice Loss for segmentation.
    Better than BCE for imbalanced classes (much more background than text).
    """
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, target):
        # Apply sigmoid to predictions
        pred = torch.sigmoid(pred)

        # Flatten
        pred = pred.view(-1)
        target = target.view(-1)

        # Dice coefficient
        intersection = (pred * target).sum()
        dice = (2. * intersection + self.smooth) / (pred.sum() + target.sum() + self.smooth)

        # Dice loss
        return 1 - dice


class CombinedLoss(nn.Module):
    """
    Combination of BCE and Dice Loss.
    BCE helps with pixel-wise accuracy, Dice helps with overall overlap.
    """
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(self, pred, target):
        return self.bce(pred, target) + self.dice(pred, target)


def save_model_fp16(model, output_dir):
    """
    Save model in fp16 format to reduce file size.
    Converts model to fp16, saves, then converts back to fp32 for continued training.
    """
    # Convert to fp16
    model_fp16 = model.half()

    # Save
    model_fp16.save_pretrained(output_dir)

    # Convert back to fp32 for training
    model.float()


def train_one_epoch(model, dataloader, optimizer, criterion, device, epoch):
    """Train for one epoch."""
    model.train()
    epoch_loss = 0.0

    progress = tqdm(dataloader, desc=f"Epoch {epoch+1}/{Config.EPOCHS}")

    for batch_idx, (images, text_masks, affinity_masks, img_names) in enumerate(progress):
        # Move to device
        images = images.to(device)
        text_masks = text_masks.to(device)
        affinity_masks = affinity_masks.to(device)

        # Forward pass
        outputs = model(pixel_values=images)
        logits = outputs.logits  # Shape: [B, 2, H, W]

        # Interpolate logits to match mask size if needed
        # The model may output at a different resolution (e.g., 128x128 instead of 512x512)
        if logits.shape[2:] != text_masks.shape[1:]:
            logits = F.interpolate(
                logits,
                size=text_masks.shape[1:],  # Target size (512, 512)
                mode="bilinear",
                align_corners=False
            )

        # Extract text and affinity predictions
        text_pred = logits[:, 0, :, :]      # Text channel
        affinity_pred = logits[:, 1, :, :]  # Affinity channel

        # Compute losses
        text_loss = criterion(text_pred, text_masks)

        # Affinity loss (optional, weight is 0 by default)
        if Config.AFFINITY_LOSS_WEIGHT > 0:
            affinity_loss = criterion(affinity_pred, affinity_masks)
            total_loss = (Config.TEXT_LOSS_WEIGHT * text_loss +
                         Config.AFFINITY_LOSS_WEIGHT * affinity_loss)
        else:
            total_loss = text_loss

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # Update progress
        epoch_loss += total_loss.item()
        progress.set_postfix({
            'loss': total_loss.item(),
            'avg_loss': epoch_loss / (batch_idx + 1)
        })

    avg_loss = epoch_loss / len(dataloader)
    return avg_loss


def main():
    print("="*60)
    print("Simple Fine-tuning for Surya Text Detection")
    print("="*60)
    print(f"Device: {Config.DEVICE}")
    print(f"Batch size: {Config.BATCH_SIZE}")
    print(f"Learning rate: {Config.LEARNING_RATE}")
    print(f"Epochs: {Config.EPOCHS}")
    print()

    # Check if dataset exists
    if not os.path.exists(Config.IMAGES_DIR):
        print(f"Error: Images directory not found: {Config.IMAGES_DIR}")
        print("Please create the dataset with structure:")
        print("  data/images/  - containing your images")
        print("  data/labels/  - containing YOLO format labels")
        return

    # Create output directory
    os.makedirs(Config.OUTPUT_DIR, exist_ok=True)

    # ========================================================================
    # 1. Load pretrained model
    # ========================================================================
    print("Loading pretrained Surya model...")

    # Load model in fp32 for training (more stable)
    # Will be saved in fp16 at the end to reduce size
    config = EfficientViTConfig.from_pretrained(Config.PRETRAINED_MODEL)
    model = EfficientViTForSemanticSegmentation.from_pretrained(
        Config.PRETRAINED_MODEL,
        config=config,
        torch_dtype=torch.float32  # Always train in fp32
    )
    model = model.to(Config.DEVICE)

    # ========================================================================
    # IMPORTANT: With small datasets (<1000 images), freeze the encoder
    # Only fine-tune the decoder head to avoid destroying pretrained features
    # ========================================================================
    total_params = sum(p.numel() for p in model.parameters())

    # Check if dataset is small
    if len(os.listdir(Config.IMAGES_DIR)) < 1000:
        print("⚠️  Small dataset detected (<1000 images)")
        print("   Freezing encoder, fine-tuning decoder only...")

        # Freeze encoder (EfficientViT backbone)
        for name, param in model.named_parameters():
            if "decode_head" not in name:
                param.requires_grad = False

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    print()

    # ========================================================================
    # 2. Create dataset and dataloader
    # ========================================================================
    print("Loading dataset...")
    dataset = YOLOTextDataset(
        images_dir=Config.IMAGES_DIR,
        labels_dir=Config.LABELS_DIR,
        image_size=Config.IMAGE_SIZE
    )

    dataloader = DataLoader(
        dataset,
        batch_size=Config.BATCH_SIZE,
        shuffle=True,
        num_workers=Config.NUM_WORKERS,
        pin_memory=True if Config.DEVICE == "cuda" else False
    )
    print(f"Dataset size: {len(dataset)} images")
    print(f"Batches per epoch: {len(dataloader)}")
    print()

    # ========================================================================
    # 3. Setup optimizer and loss
    # ========================================================================
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=Config.LEARNING_RATE,
        weight_decay=Config.WEIGHT_DECAY
    )

    # Use combined loss (BCE + Dice)
    criterion = CombinedLoss()

    # Optional: Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=Config.EPOCHS
    )

    print("Optimizer: AdamW")
    print("Loss: Combined (BCE + Dice)")
    print()

    # ========================================================================
    # 4. Training loop
    # ========================================================================
    print("Starting training...")
    print("="*60)

    best_loss = float('inf')

    for epoch in range(Config.EPOCHS):
        # Train one epoch
        avg_loss = train_one_epoch(
            model, dataloader, optimizer, criterion, Config.DEVICE, epoch
        )

        # Update learning rate
        scheduler.step()

        print(f"\nEpoch {epoch+1}/{Config.EPOCHS}")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Learning Rate: {scheduler.get_last_lr()[0]:.2e}")

        # Save checkpoint
        if (epoch + 1) % Config.SAVE_EVERY == 0:
            checkpoint_dir = os.path.join(Config.OUTPUT_DIR, f"checkpoint_epoch_{epoch+1}")
            os.makedirs(checkpoint_dir, exist_ok=True)
            save_model_fp16(model, checkpoint_dir)
            print(f"  Saved checkpoint to {checkpoint_dir}")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_dir = os.path.join(Config.OUTPUT_DIR, "best_model")
            os.makedirs(best_dir, exist_ok=True)
            save_model_fp16(model, best_dir)
            print(f"  New best model! Loss: {best_loss:.4f}")

        print()

    # ========================================================================
    # 5. Save final model
    # ========================================================================
    final_dir = os.path.join(Config.OUTPUT_DIR, "final_model")
    os.makedirs(final_dir, exist_ok=True)
    save_model_fp16(model, final_dir)

    print("="*60)
    print("Training complete!")
    print(f"Best loss: {best_loss:.4f}")
    print(f"Final model saved to: {final_dir}")
    print(f"Best model saved to: {os.path.join(Config.OUTPUT_DIR, 'best_model')}")
    print("="*60)
    print()
    print("To use your fine-tuned model with simple_inf.py:")
    print(f"  1. Replace MODEL_CHECKPOINT with: '{final_dir}'")
    print("  2. Run: python simple_inf.py")


# ============================================================================
# VISUALIZATION (optional utility)
# ============================================================================

def visualize_batch(images, masks, predictions=None, save_path=None):
    """
    Visualize a batch of images with their masks and predictions.
    Useful for debugging.
    """
    import matplotlib.pyplot as plt

    batch_size = min(4, images.shape[0])
    fig, axes = plt.subplots(batch_size, 3, figsize=(12, 4*batch_size))

    for i in range(batch_size):
        # Denormalize image
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        img = img * Config.STD + Config.MEAN
        img = np.clip(img, 0, 1)

        # Get mask
        mask = masks[i].cpu().numpy()

        # Plot image
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"Image {i+1}")
        axes[i, 0].axis('off')

        # Plot ground truth mask
        axes[i, 1].imshow(mask, cmap='gray')
        axes[i, 1].set_title(f"Ground Truth")
        axes[i, 1].axis('off')

        # Plot prediction if available
        if predictions is not None:
            pred = torch.sigmoid(predictions[i, 0]).cpu().numpy()
            axes[i, 2].imshow(pred, cmap='gray')
            axes[i, 2].set_title(f"Prediction")
        else:
            axes[i, 2].axis('off')

        axes[i, 2].axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved visualization to {save_path}")
    else:
        plt.show()

    plt.close()


if __name__ == "__main__":
    main()

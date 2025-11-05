#!/usr/bin/env python3
"""
Simple inference script for Surya line detection model.
Reimplements the inference pipeline from scratch without using surya's helper functions.
Only uses the model architecture from surya, everything else is pure PyTorch/CV2.
"""

import os
import cv2
import torch
import numpy as np
from PIL import Image, ImageDraw
import torch.nn.functional as F

# Import only the model architecture and config from surya (not the inference pipeline)
from surya.detection.model.config import EfficientViTConfig
from surya.detection.model.encoderdecoder import EfficientViTForSemanticSegmentation


# ============================================================================
# MODEL DOWNLOAD & LOADING
# ============================================================================

MODEL_CHECKPOINT = "s3://text_detection/2025_05_07"
S3_BASE_URL = "https://models.datalab.to"
CACHE_DIR = os.path.expanduser("~/.cache/datalab/models")


def load_model():
    """Load the detection model from checkpoint."""
    print("Loading model...")

    # Load config and model using from_pretrained (auto-downloads from S3)
    config = EfficientViTConfig.from_pretrained(MODEL_CHECKPOINT)
    model = EfficientViTForSemanticSegmentation.from_pretrained(
        MODEL_CHECKPOINT,
        config=config,
    )

    # Set to eval mode
    model.eval()

    # Move to CPU (change to 'cuda' if you have GPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)

    print(f"Model loaded on {device}")
    return model, device


# ============================================================================
# IMAGE PREPROCESSING
# ============================================================================

# ImageNet normalization stats (used by SegformerImageProcessor)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])
TARGET_SIZE = (512, 512)  # Model expects 512x512 input


def preprocess_image(image_path):
    """
    Preprocess image for model inference.
    Replicates the preprocessing done in surya.detection.DetectionPredictor.prepare_image
    """
    # Load image
    img = Image.open(image_path).convert("RGB")
    orig_size = img.size  # (width, height)

    print(f"Original image size: {orig_size}")

    # Resize to 512x512 (double resize for accuracy as in surya)
    img.thumbnail(TARGET_SIZE, Image.Resampling.LANCZOS)
    img = img.resize(TARGET_SIZE, Image.Resampling.LANCZOS)

    # Convert to numpy array [H, W, C]
    img_np = np.asarray(img, dtype=np.float32)

    # Rescale from [0, 255] to [0, 1]
    img_np = img_np / 255.0

    # Normalize with ImageNet stats
    img_np = (img_np - IMAGENET_MEAN) / IMAGENET_STD

    # Convert to [C, H, W] format
    img_np = np.transpose(img_np, (2, 0, 1))

    # Convert to torch tensor and add batch dimension [1, C, H, W]
    img_tensor = torch.from_numpy(img_np).unsqueeze(0).float()

    return img_tensor, orig_size


# ============================================================================
# MODEL INFERENCE
# ============================================================================

def run_inference(model, img_tensor, device):
    """Run model inference to get heatmaps."""
    print("Running inference...")

    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        outputs = model(pixel_values=img_tensor)

    # Get logits [1, 2, H, W] - 2 channels: text and affinity
    logits = outputs.logits

    # Interpolate to ensure correct size (512x512)
    if logits.shape[2:] != (512, 512):
        logits = F.interpolate(logits, size=(512, 512), mode="bilinear", align_corners=False)

    # Convert to numpy [H, W] for text channel (channel 0)
    heatmap = logits[0, 0].cpu().numpy()

    print(f"Heatmap shape: {heatmap.shape}, range: [{heatmap.min():.3f}, {heatmap.max():.3f}]")

    return heatmap


# ============================================================================
# POSTPROCESSING - BOX DETECTION
# ============================================================================

def get_dynamic_thresholds(heatmap, text_threshold=0.6, low_text=0.35):
    """
    Adjust thresholds dynamically based on image statistics.
    Replicates surya.detection.heatmap.get_dynamic_thresholds
    """
    # Find average intensity of top 10% pixels
    flat_map = heatmap.ravel()
    top_10_count = int(len(flat_map) * 0.9)
    avg_intensity = np.mean(np.partition(flat_map, top_10_count)[top_10_count:])

    typical_top10_avg = 0.7
    scaling_factor = np.clip(avg_intensity / typical_top10_avg, 0, 1) ** 0.5

    low_text = np.clip(low_text * scaling_factor, 0.1, 0.6)
    text_threshold = np.clip(text_threshold * scaling_factor, 0.15, 0.8)

    print(f"Dynamic thresholds: text={text_threshold:.3f}, low_text={low_text:.3f}")

    return text_threshold, low_text


def detect_boxes(heatmap, text_threshold=0.6, low_text=0.35):
    """
    Detect bounding boxes from heatmap using connected components.
    Replicates surya.detection.heatmap.detect_boxes
    """
    img_h, img_w = heatmap.shape

    # Adjust thresholds dynamically
    text_threshold, low_text = get_dynamic_thresholds(heatmap, text_threshold, low_text)

    # Create binary mask
    text_score_comb = (heatmap > low_text).astype(np.uint8)

    # Find connected components
    label_count, labels, stats, centroids = cv2.connectedComponentsWithStats(
        text_score_comb, connectivity=4
    )

    print(f"Found {label_count - 1} connected components")

    boxes = []
    confidences = []
    max_confidence = 0

    for k in range(1, label_count):  # Skip background (0)
        # Filter by size
        size = stats[k, cv2.CC_STAT_AREA]
        if size < 10:
            continue

        # Get bounding box stats
        x, y, w, h = stats[k, [cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP,
                               cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT]]

        # Dilate the component
        niter = int(np.sqrt(min(w, h))) if min(w, h) > 0 else 0
        buffer = 1
        sx, sy = max(0, x - niter - buffer), max(0, y - niter - buffer)
        ex, ey = min(img_w, x + w + niter + buffer), min(img_h, y + h + niter + buffer)

        # Get mask for this component
        mask = labels[sy:ey, sx:ex] == k
        selected_heatmap = heatmap[sy:ey, sx:ex][mask]

        if selected_heatmap.size == 0:
            continue

        line_max = np.max(selected_heatmap)

        # Threshold check
        if line_max < text_threshold:
            continue

        # Dilate mask
        segmap = mask.astype(np.uint8)
        ksize = buffer + niter
        if ksize > 0:
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
            segmap = cv2.dilate(segmap, kernel)

        # Fit rotated rectangle to component
        y_inds, x_inds = np.nonzero(segmap)
        if len(x_inds) < 5:  # Need at least 5 points for minAreaRect
            continue

        x_inds += sx
        y_inds += sy

        np_contours = np.column_stack((x_inds, y_inds))
        rectangle = cv2.minAreaRect(np_contours)
        box = cv2.boxPoints(rectangle)

        # Align diamond-shaped boxes to rectangles
        w_box, h_box = np.linalg.norm(box[0] - box[1]), np.linalg.norm(box[1] - box[2])
        box_ratio = max(w_box, h_box) / (min(w_box, h_box) + 1e-5)

        if abs(1 - box_ratio) <= 0.1:  # Nearly square
            left, right = np_contours[:, 0].min(), np_contours[:, 0].max()
            top, bottom = np_contours[:, 1].min(), np_contours[:, 1].max()
            box = np.array([[left, top], [right, top], [right, bottom], [left, bottom]],
                          dtype=np.float32)

        # Make clockwise order (top-left first)
        startidx = box.sum(axis=1).argmin()
        box = np.roll(box, 4 - startidx, 0)

        max_confidence = max(max_confidence, line_max)
        confidences.append(line_max)
        boxes.append(box)

    # Normalize confidences
    if max_confidence > 0:
        confidences = [c / max_confidence for c in confidences]

    print(f"Detected {len(boxes)} boxes after filtering")

    return boxes, confidences


def rescale_boxes(boxes, from_size=(512, 512), to_size=None):
    """Rescale boxes from model size to original image size."""
    if to_size is None:
        return boxes

    width_scale = to_size[0] / from_size[0]
    height_scale = to_size[1] / from_size[1]

    rescaled_boxes = []
    for box in boxes:
        rescaled_box = box.copy()
        rescaled_box[:, 0] *= width_scale
        rescaled_box[:, 1] *= height_scale
        rescaled_boxes.append(rescaled_box.astype(int))

    return rescaled_boxes


# ============================================================================
# VISUALIZATION
# ============================================================================

def draw_boxes_on_image(image_path, boxes, output_path="out.png"):
    """Draw detected boxes on image and save."""
    print(f"Drawing {len(boxes)} boxes on image...")

    # Load original image
    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    # Draw each box
    for box in boxes:
        # Convert to list of tuples for polygon
        poly = [(int(p[0]), int(p[1])) for p in box]
        draw.polygon(poly, outline="red", width=2)

    # Save result
    img.save(output_path)
    print(f"Result saved to {output_path}")

    return img


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    # Input image
    image_path = "test.png"

    if not os.path.exists(image_path):
        print(f"Error: {image_path} not found!")
        print("Please make sure test.png exists in the current directory.")
        return

    # 1. Load model
    model, device = load_model()

    # 2. Preprocess image
    img_tensor, orig_size = preprocess_image(image_path)

    # 3. Run inference
    heatmap = run_inference(model, img_tensor, device)

    # 4. Detect boxes from heatmap
    boxes, confidences = detect_boxes(heatmap)

    # 5. Rescale boxes to original image size
    boxes = rescale_boxes(boxes, from_size=(512, 512), to_size=orig_size)

    # 6. Draw boxes and save
    result_img = draw_boxes_on_image(image_path, boxes, output_path="out.png")

    print("\n" + "="*60)
    print("DONE! Check out.png for the result.")
    print(f"Detected {len(boxes)} text lines")
    print("="*60)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Face redaction script for NATHAN MATTHIS FILES.
Reads images from input folder, detects faces using OpenCV Haar cascades,
draws black rectangles over detected faces, saves to output folder.
"""

import cv2
import os
import sys
import glob
import base64
import json

def redact_faces(input_dir, output_dir):
    """Detect and redact faces in all images in input_dir, save to output_dir."""
    os.makedirs(output_dir, exist_ok=True)

    # Load multiple cascade classifiers for better detection
    cascade_paths = [
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml',
        cv2.data.haarcascades + 'haarcascade_frontalface_alt.xml',
        cv2.data.haarcascades + 'haarcascade_frontalface_alt2.xml',
        cv2.data.haarcascades + 'haarcascade_profileface.xml',
    ]
    
    cascades = []
    for p in cascade_paths:
        c = cv2.CascadeClassifier(p)
        if not c.empty():
            cascades.append(c)
    
    if not cascades:
        print("ERROR: No cascade classifiers loaded!", file=sys.stderr)
        sys.exit(1)

    # Supported image extensions
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp', '*.JPG', '*.JPEG', '*.PNG']
    
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
    
    image_files = sorted(set(image_files))
    
    if not image_files:
        print(f"No images found in {input_dir}")
        sys.exit(1)

    results = []

    for img_path in image_files:
        filename = os.path.basename(img_path)
        print(f"Processing: {filename}")
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"  WARNING: Could not read {filename}, skipping.")
            continue
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        
        all_faces = []
        
        # Try each cascade at multiple scales
        for cascade in cascades:
            faces = cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(30, 30),
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            if len(faces) > 0:
                all_faces.extend(faces.tolist())
        
        # Also try on a larger version for small faces
        h, w = gray.shape
        if max(h, w) < 1000:
            scale = 1000 / max(h, w)
            gray_large = cv2.resize(gray, None, fx=scale, fy=scale)
            for cascade in cascades:
                faces = cascade.detectMultiScale(
                    gray_large,
                    scaleFactor=1.1,
                    minNeighbors=4,
                    minSize=(30, 30),
                    flags=cv2.CASCADE_SCALE_IMAGE
                )
                if len(faces) > 0:
                    for (x, y, fw, fh) in faces:
                        all_faces.append([int(x/scale), int(y/scale), int(fw/scale), int(fh/scale)])
        
        # Merge overlapping detections (non-maximum suppression)
        merged = merge_detections(all_faces)
        
        print(f"  Found {len(merged)} face(s)")
        
        # Draw black rectangles over faces with slight padding
        for (x, y, fw, fh) in merged:
            pad_x = int(fw * 0.1)
            pad_y = int(fh * 0.1)
            x1 = max(0, x - pad_x)
            y1 = max(0, y - pad_y)
            x2 = min(img.shape[1], x + fw + pad_x)
            y2 = min(img.shape[0], y + fh + pad_y)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), -1)
        
        out_path = os.path.join(output_dir, filename)
        cv2.imwrite(out_path, img)
        print(f"  Saved: {out_path}")
        
        results.append({
            "filename": filename,
            "faces_found": len(merged),
            "path": out_path
        })
    
    # Output JSON summary
    print("\n--- RESULTS ---")
    print(json.dumps(results, indent=2))
    return results


def merge_detections(detections):
    """Merge overlapping face detections using IoU."""
    if not detections:
        return []
    
    boxes = [(x, y, x + w, y + h) for (x, y, w, h) in detections]
    merged = []
    used = [False] * len(boxes)
    
    for i in range(len(boxes)):
        if used[i]:
            continue
        group = [boxes[i]]
        used[i] = True
        for j in range(i + 1, len(boxes)):
            if used[j]:
                continue
            if iou(boxes[i], boxes[j]) > 0.3:
                group.append(boxes[j])
                used[j] = True
        # Average the group
        x1 = int(sum(b[0] for b in group) / len(group))
        y1 = int(sum(b[1] for b in group) / len(group))
        x2 = int(sum(b[2] for b in group) / len(group))
        y2 = int(sum(b[3] for b in group) / len(group))
        merged.append((x1, y1, x2 - x1, y2 - y1))
    
    return merged


def iou(box1, box2):
    """Intersection over Union for two boxes (x1, y1, x2, y2)."""
    xi1 = max(box1[0], box2[0])
    yi1 = max(box1[1], box2[1])
    xi2 = min(box1[2], box2[2])
    yi2 = min(box1[3], box2[3])
    inter = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


if __name__ == "__main__":
    input_dir = sys.argv[1] if len(sys.argv) > 1 else "./photos"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./redacted"
    redact_faces(input_dir, output_dir)

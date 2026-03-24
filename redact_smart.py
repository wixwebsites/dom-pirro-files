#!/usr/bin/env python3
"""
Face redaction that keeps one person's face visible.
Uses OpenCV Haar cascades for detection + histogram comparison to match the reference face.
"""

import cv2
import numpy as np
import os
import sys
import glob
import json

def get_face_encoding(face_img):
    """Get a comparable encoding of a face using color histogram + structure."""
    if face_img is None or face_img.size == 0:
        return None
    face_resized = cv2.resize(face_img, (100, 100))
    hsv = cv2.cvtColor(face_resized, cv2.COLOR_BGR2HSV)
    # Hue-Saturation histogram
    hist_hs = cv2.calcHist([hsv], [0, 1], None, [30, 32], [0, 180, 0, 256])
    cv2.normalize(hist_hs, hist_hs)
    # Grayscale structure via LBP-like features
    gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    hist_gray = cv2.calcHist([gray], [0], None, [64], [0, 256])
    cv2.normalize(hist_gray, hist_gray)
    return (hist_hs, hist_gray)

def compare_faces(enc1, enc2):
    """Compare two face encodings. Higher = more similar."""
    if enc1 is None or enc2 is None:
        return 0
    hs_score = cv2.compareHist(enc1[0], enc2[0], cv2.HISTCMP_CORREL)
    gray_score = cv2.compareHist(enc1[1], enc2[1], cv2.HISTCMP_CORREL)
    return hs_score * 0.6 + gray_score * 0.4

def detect_faces(img, cascades):
    """Detect faces using multiple cascades."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    all_faces = []
    
    for cascade in cascades:
        for scale in [1.05, 1.1, 1.15]:
            for min_n in [3, 4, 5]:
                faces = cascade.detectMultiScale(
                    gray, scaleFactor=scale, minNeighbors=min_n,
                    minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE
                )
                if len(faces) > 0:
                    all_faces.extend(faces.tolist())
    
    # Also try on upscaled version for small faces
    h, w = gray.shape
    if max(h, w) < 1200:
        scale_factor = 1200 / max(h, w)
        gray_large = cv2.resize(gray, None, fx=scale_factor, fy=scale_factor)
        for cascade in cascades:
            faces = cascade.detectMultiScale(
                gray_large, scaleFactor=1.1, minNeighbors=4,
                minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE
            )
            if len(faces) > 0:
                for (x, y, fw, fh) in faces:
                    all_faces.append([
                        int(x/scale_factor), int(y/scale_factor),
                        int(fw/scale_factor), int(fh/scale_factor)
                    ])
    
    return merge_detections(all_faces)

def merge_detections(detections):
    """Merge overlapping detections."""
    if not detections:
        return []
    boxes = [(x, y, x+w, y+h) for (x, y, w, h) in detections]
    merged = []
    used = [False] * len(boxes)
    for i in range(len(boxes)):
        if used[i]: continue
        group = [boxes[i]]
        used[i] = True
        for j in range(i+1, len(boxes)):
            if used[j]: continue
            if iou(boxes[i], boxes[j]) > 0.3:
                group.append(boxes[j])
                used[j] = True
        x1 = int(sum(b[0] for b in group) / len(group))
        y1 = int(sum(b[1] for b in group) / len(group))
        x2 = int(sum(b[2] for b in group) / len(group))
        y2 = int(sum(b[3] for b in group) / len(group))
        merged.append((x1, y1, x2-x1, y2-y1))
    return merged

def iou(b1, b2):
    xi1, yi1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    xi2, yi2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0, xi2-xi1) * max(0, yi2-yi1)
    a1 = (b1[2]-b1[0]) * (b1[3]-b1[1])
    a2 = (b2[2]-b2[0]) * (b2[3]-b2[1])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0

def main():
    ref_path = sys.argv[1] if len(sys.argv) > 1 else "./main.png.png"
    input_dir = sys.argv[2] if len(sys.argv) > 2 else "./photos"
    output_dir = sys.argv[3] if len(sys.argv) > 3 else "./redacted"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Load cascades
    cascade_files = [
        'haarcascade_frontalface_default.xml',
        'haarcascade_frontalface_alt.xml',
        'haarcascade_frontalface_alt2.xml',
        'haarcascade_profileface.xml',
    ]
    cascades = []
    for cf in cascade_files:
        c = cv2.CascadeClassifier(cv2.data.haarcascades + cf)
        if not c.empty():
            cascades.append(c)
    
    # Get reference face encoding
    ref_img = cv2.imread(ref_path)
    if ref_img is None:
        print(f"ERROR: Cannot read reference image: {ref_path}")
        sys.exit(1)
    
    # Detect face in reference image
    ref_faces = detect_faces(ref_img, cascades)
    if not ref_faces:
        print("WARNING: No face detected in reference image, using whole image as reference")
        ref_encoding = get_face_encoding(ref_img)
    else:
        # Use the largest face in reference
        ref_faces.sort(key=lambda f: f[2]*f[3], reverse=True)
        rx, ry, rw, rh = ref_faces[0]
        ref_crop = ref_img[ry:ry+rh, rx:rx+rw]
        ref_encoding = get_face_encoding(ref_crop)
        print(f"Reference face detected: {rw}x{rh}")
    
    # Process photos
    extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp',
                  '*.JPG', '*.JPEG', '*.PNG']
    image_files = []
    for ext in extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
    image_files = sorted(set(image_files))
    
    # Skip non-image files
    valid_files = []
    for f in image_files:
        img = cv2.imread(f)
        if img is not None and img.size > 100:
            valid_files.append(f)
        else:
            print(f"SKIP (not a valid image): {os.path.basename(f)}")
    
    results = []
    for img_path in valid_files:
        filename = os.path.basename(img_path)
        print(f"\nProcessing: {filename}")
        
        img = cv2.imread(img_path)
        faces = detect_faces(img, cascades)
        print(f"  Detected {len(faces)} face(s)")
        
        kept = 0
        redacted = 0
        
        for (x, y, w, h) in faces:
            face_crop = img[y:y+h, x:x+w]
            face_enc = get_face_encoding(face_crop)
            similarity = compare_faces(ref_encoding, face_enc)
            
            print(f"  Face at ({x},{y}) {w}x{h}: similarity={similarity:.3f}")
            
            # Threshold: if similar enough to reference, keep it
            if similarity > 0.35:
                print(f"    -> KEEP (matches reference)")
                kept += 1
            else:
                print(f"    -> REDACT")
                pad_x = int(w * 0.1)
                pad_y = int(h * 0.1)
                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(img.shape[1], x + w + pad_x)
                y2 = min(img.shape[0], y + h + pad_y)
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 0), -1)
                redacted += 1
        
        out_path = os.path.join(output_dir, filename)
        cv2.imwrite(out_path, img)
        print(f"  Result: kept={kept}, redacted={redacted}")
        results.append({"filename": filename, "kept": kept, "redacted": redacted})
    
    print("\n--- RESULTS ---")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()

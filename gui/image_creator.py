#######################################
### image_creator.py  Version Greece
#######################################

import os
import sys
import json
from PIL import Image, ImageTk
import tkinter as tk
#import cv2
import numpy as np
import math

c_default_customerimage_name = "customerimage"
margin_ratio =0.3

class ImageCreator:
    def __init__(self, working_directory, pad_to_square=None):
        self.working_directory = working_directory
        self.pad_to_square = pad_to_square

    def enhance_face_image(self, pil_image):
        # Ensure input is PIL
        if not isinstance(pil_image, Image.Image):
            raise TypeError("Input must be a PIL Image.")

        # Assess quality (handles PIL internally)
        result = assess_quality(pil_image)
        score = result["brisque_score"]
        sharpness = result["sharpness_score"]

        # Convert to OpenCV format for enhancement
        img_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        # Apply enhancement based on score
        if score > 40:
            enhanced = full_enhance(img_cv)
        elif score > 30:
            enhanced = light_enhance(img_cv)
        else:
            enhanced = no_enhance(img_cv)

        print(result)

        # Convert back to PIL
        enhanced_pil = Image.fromarray(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
        return enhanced_pil

    def display_all(self, json_path):
        results = {
            "intent": None,
            "main_image": None,
            "faces": [],
            "animals": [],
            "objects": [],
            "annotated_image": None,
            "couple": False,
            "coupleimage": None
        }

        # Check if JSON exists
        if not os.path.exists(json_path):
            print(f"[ERROR] JSON file not found: {json_path}")
            return results

        # Load metadata
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                metadata = json.load(f)
        except Exception as e:
            print(f"[ERROR] Reading JSON: {e}")
            return results

        def resolve_path(path_key):
            path = metadata.get(path_key)
            if not path: return None
            return path if os.path.isabs(path) else os.path.join(self.working_directory, path)

        # Load annotated image
        annotated_path = resolve_path("annotated_image")
        if annotated_path:
            try:
                img = Image.open(annotated_path).convert("RGB")
                img.thumbnail((250, 250), Image.Resampling.LANCZOS)
                results["main_image"] = img
            except Exception as e:
                print(f"[ERROR] Loading annotated image: {e}")

        results["intent"] = metadata.get("intent", "Unknown")

        # Load original image
        original_path = resolve_path("original_image_name")
        if not original_path:
            print("[ERROR] Original image path missing.")
            return results

        try:
            o_img = Image.open(original_path).convert("RGB")
        except Exception as e:
            print(f"[ERROR] Loading original image: {e}")
            return results

        # Prepare face data
        results["faces"] = metadata.get("faces", [])
        results["animals"] = metadata.get("animals", [])
        results["objects"] = metadata.get("objects", [])
        results["couple"] = metadata.get("couple", [])
        results["annotated_image"] = metadata.get("annotated_image", [])
        results["original_image_name"] = metadata.get("original_image_name", [])
        results["total_faces"] = metadata.get("total_faces", [])
        results["total_animals"] = metadata.get("total_animals", [])
        results["coupleimage"] = metadata.get("coupleimage", [])

        return results

def resize_with_aspect(image, target_size=(256, 256)):
    # Convert PIL to OpenCV format if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    h, w = image.shape[:2]
    scale = min(target_size[0] / h, target_size[1] / w)
    new_w, new_h = int(w * scale), int(h * scale)

    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    canvas = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)

    x_offset = (target_size[1] - new_w) // 2
    y_offset = (target_size[0] - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

    # Convert back to PIL format (RGB)
    canvas_rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    return Image.fromarray(canvas_rgb)

def detect_blur(image):
    # Check if image is a PIL Image
    if isinstance(image, Image.Image):
        # Convert PIL to NumPy array (RGB)
        image = np.array(image)
        # Convert RGB to BGR (OpenCV uses BGR)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Proceed with blur detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lap_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return lap_var

def assess_quality(image):
    # Validate input
    if image is None:
        raise ValueError("Invalid image input: None provided.")

    # Convert PIL to OpenCV format if needed
    if isinstance(image, Image.Image):
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Sharpness score (works with OpenCV image)
    sharpness_score = detect_blur(image)  # Sehr scharf > 1000, unscharf < 100

    # Resize and convert back to OpenCV format for BRISQUE
    resized_pil = resize_with_aspect(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), target_size=(256, 256))
    resized_cv = cv2.cvtColor(np.array(resized_pil), cv2.COLOR_RGB2BGR)

    # Ensure correct format
    resized_cv = resized_cv.astype("uint8")

    # Compute BRISQUE score
    brisque_score = brisque.compute(resized_cv)[0]
    print("Brisque score from assess_quality:", brisque_score)

    return {
        "brisque_score": brisque_score,
        "sharpness_score": sharpness_score
    }

def full_enhance(img):
    print("Type of img passed to full_enhance:", type(img))

    if isinstance(img, np.ndarray):
        # If it's BGR (from OpenCV), convert to RGB first
        if img.shape[-1] == 3:  # Assuming color image
            img = Image.fromarray(img[..., ::-1])  # BGR to RGB
        else:
            img = Image.fromarray(img)  # Grayscale or already RGB
    if not isinstance(img, Image.Image):
        raise TypeError("Expected a PIL image")
    print("Full enhance")

    # Convert PIL image to NumPy array (RGB → BGR)
    img_np = np.array(img)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

    # Upscale image (simulate super-resolution)
    upscale = cv2.resize(img_bgr, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Apply bilateral filter to smooth skin while preserving edges
    smooth = cv2.bilateralFilter(upscale, d=9, sigmaColor=75, sigmaSpace=75)

    # Boost contrast slightly
    lab = cv2.cvtColor(smooth, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced = cv2.merge((cl, a, b))
    final_bgr = cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)

    # Optional debug label
    final_bgr = add_debug_label(final_bgr, label="F")

    # Convert back to PIL (BGR → RGB)
    final_rgb = cv2.cvtColor(final_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(final_rgb)

def light_enhance(img):
    print("light enhance")
    # Apply mild sharpening
    kernel = np.array([[0, -1, 0],
                       [-1, 5, -1],
                       [0, -1, 0]])
    sharpened = cv2.filter2D(img, -1, kernel)
    # Slight brightness and contrast adjustment
    adjusted = cv2.convertScaleAbs(sharpened, alpha=1.1, beta=10)
    adjusted = add_debug_label(adjusted, label="L")
    return adjusted

def no_enhance(img):
    print("no enhance")
    img = add_debug_label(img, label="0")
    return img

def make_square_box(x1, y1, x2, y2):
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    w = x2 - x1
    h = y2 - y1
    side = max(w, h)
    half_side = side // 2
    new_x1 = cx - half_side
    new_y1 = cy - half_side
    new_x2 = cx + half_side
    new_y2 = cy + half_side
    return new_x1, new_y1, new_x2, new_y2

def add_debug_label(img, label="F"):
    height = img.shape[0]
    font_scale = height / 3 / 30  # Adjust divisor to fine-tune size
    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = max(1, int(font_scale))  # Ensure visible stroke
    position = (10, int(height * 0.9))   # Bottom-left corner
    cv2.putText(img, label, position, font, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)
    return img



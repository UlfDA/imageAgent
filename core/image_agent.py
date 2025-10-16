##############################################################################
### image_agent.py  Version :Phillip1: 10.9.2025
##############################################################################
c_default_customerimage_name = "customerimage"
c_default_coupleimage_name = "coupleimage"
c_default_customerimage_name_no_Background = "customerimage_no_background"
c_default_customerimage_name_grc = "customerimage_grc"    #g= gray ; r= removed background ; c= clahe contrastes
c_default_customerimage_name_grc_nobgd = "customerimage_grc_nobgrd"   #
json_dateiname = "MetadataAI.json"
default_scalesize = 1000

import os
import sys
sys.path.append("C:/Users/unbehaun/PycharmProjects/removebg")

from image_tools import photogift_composer
import time
startAgent = time.time()        # zum debuggen der Laufzeiten der Classifikatoren; Auch die Ladezeiten sind beträchtlich; kann aber alles auch gelöscht werden
from typing import List, Dict, Tuple, Union
from PIL import Image, ImageDraw,ImageFont
import numpy as np
import math
#import insightface   # 10 s
from insightface.app import FaceAnalysis   # für die Gesichtserkennung allgemein. Schlecht in der Alters und Geschlechtsbestimmung
import json
import shutil
#import clip
#import torch     # core deep learning engine -> like numpy but GPU; Neural networks; a powerful open-source framework for deep learning and tensor computation. Once imported, you can use PyTorch to Create and manipulate tensors (multi-dimensional arrays, like NumPy but GPU-ready) Build and train neural networks Use pre-trained models and tools for computer vision, NLP, and more Move computations to GPU (if available) for faster performanceLädt in 4 s ; a powerful open-source framework for deep learning and tensor computation.
#from torchvision import transforms  #  image preprocessing utilities for vision tasks
#from facenet_pytorch import MTCNN    # Für Alter und Geschlecht wird fairface benutzt. Das Model kommt direkt über Huggingface
#from ultralytics import YOLO
#from transformers import pipeline   # Load the age classification pipeline
#from transformers import AutoModelForImageSegmentation
print(f"Agent: imports: {time.time() - startAgent:.2f} seconds")

from image_tools import remove_background
from image_tools import clahe_contrast
from image_tools import convert_to_grayscale
from image_tools import stretch_grayscale
from image_tools import unsharp_mask
### Die image _tool laden in 6 Sekunden
#print(f"Agent: nach image_tools import: {time.time() - startAgent:.2f} seconds")


margin_ratio =0.3
c_couple_adaptive_margin = 0.05
#age_classifier = pipeline("image-classification", model="dima806/fairface_age_image_detection")
#gender_classifier =  pipeline("image-classification", model="dima806/fairface_gender_image_detection")

def apply_alpha_mask(grc_path, no_bg_path, output_path):
    """
    Combines the contrast-enhanced image with the alpha mask from the no-background image.

    Parameters:
    - grc_path: Path to the contrast-enhanced image (with background).
    - no_bg_path: Path to the image with alpha mask (no background).
    - output_path: Path to save the resulting image.
    """
    # Load both images as RGBA
    no_bg = Image.open(no_bg_path).convert("RGBA")
    grc = Image.open(grc_path).convert("RGBA")

    # Extract alpha channel from the no-background image
    alpha_mask = no_bg.getchannel("A")

    # Apply the alpha mask to the contrast-enhanced image
    grc.putalpha(alpha_mask)

    # Save the result
    grc.save(output_path)

class AIModelManager:
    def __init__(self):

        self.mtcnn = None
        self.yolo_model = None
        self.age_pipeline = None
        self.segmentation_model = None

    def get_mtcnn(self):
        if self.mtcnn is None:
            from facenet_pytorch import MTCNN
            self.mtcnn = MTCNN()
        return self.mtcnn

    def get_yolo_model(self):
        if self.yolo_model is None:
            from ultralytics import YOLO
            self.yolo_model = YOLO("yolov8n.pt")
        return self.yolo_model

    def get_age_pipeline(self):
        if self.age_pipeline is None:
            from transformers import pipeline
            self.age_pipeline = pipeline("image-classification", model="fairface/fairface_age")
        return self.age_pipeline

    def get_segmentation_model(self):
        if self.segmentation_model is None:
            from transformers import AutoModelForImageSegmentation
            self.segmentation_model = AutoModelForImageSegmentation.from_pretrained("some-model-name")
        return self.segmentation_model

manager = AIModelManager()

class FairFaceManager:
    def __init__(self):
        self.age_classifier = None
        self.gender_classifier = None

    def get_age_classifier(self):
        if self.age_classifier is None:
            self.age_classifier = pipeline(
                "image-classification",
                model="dima806/fairface_age_image_detection"
            )
        return self.age_classifier

    def get_gender_classifier(self):
        if self.gender_classifier is None:
            self.gender_classifier = pipeline(
                "image-classification",
                model="dima806/fairface_gender_image_detection"
            )
        return self.gender_classifier

fairface = FairFaceManager()

class ImageAgent:
    def __init__(self, model: str = "hog"):
        self.model = model
        self.working_directory = None
        #self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device ="cpu"

        # Lazy-loaded models
        self._yolo_model = None
        self._face_app = None
        self._mtcnn = None
        self._age_classifier = None
        self._gender_classifier = None

        # Static labels
        self.age_labels = ['0-2', '3-9', '10-19', '20-29', '30-39', '40-49', '50-59', '60-69', '70+']

    def get_fairface_transform(self):
        from torchvision import transforms
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def load_torch_model(self):
        import torch
        # Example: load a model only when needed
        if self._age_classifier is None:
            self._age_classifier = torch.nn.Identity()  # Replace with actual model loading
        return self._age_classifier

    def run_inference(self, image):
        import torch
        model = self.load_torch_model()
        transform = self.get_fairface_transform()
        tensor = transform(image).unsqueeze(0)
        output = model(tensor)
        return output

    def get_face_app(self):
        if self._face_app is None:
            from insightface.app import FaceAnalysis
            self._face_app = FaceAnalysis(providers=['CPUExecutionProvider'])
            self._face_app.prepare(ctx_id=0, det_size=(640, 640))
        return self._face_app

    def get_mtcnn(self):
        if self._mtcnn is None:
            from facenet_pytorch import MTCNN
            self._mtcnn = MTCNN(keep_all=True, device=self.device)
        return self._mtcnn

    def get_age_classifier(self):
        if self._age_classifier is None:
            from transformers import pipeline
            self._age_classifier = pipeline("image-classification", model="dima806/fairface_age_image_detection")
        return self._age_classifier

    def get_gender_classifier(self):
        if self._gender_classifier is None:
            from transformers import pipeline
            self._gender_classifier = pipeline("image-classification", model="dima806/fairface_gender_image_detection")
        return self._gender_classifier

    def apply_mask_and_black_background(self,clahe_path, mask_path):
        """
        Combines CLAHE image with alpha mask from background-removed image.
        Transparent areas are filled with black. Output is RGB without alpha.
        """
        # Load CLAHE image (RGBA)
        clahe_img = Image.open(clahe_path).convert("RGBA")
        clahe_array = np.array(clahe_img)

        # Load mask image (RGBA with alpha channel)
        mask_img = Image.open(mask_path).convert("RGBA")
        mask_array = np.array(mask_img)

        # Extract alpha channel from mask
        alpha_mask = mask_array[:, :, 3] / 255.0  # Normalize to [0, 1]

        # Apply mask to CLAHE image
        r = clahe_array[:, :, 0] * alpha_mask
        g = clahe_array[:, :, 1] * alpha_mask
        b = clahe_array[:, :, 2] * alpha_mask

        # Fill transparent areas with black
        r = r.astype(np.uint8)
        g = g.astype(np.uint8)
        b = b.astype(np.uint8)

        # Merge into RGB image
        final_array = np.stack([r, g, b], axis=-1)
        final_image = Image.fromarray(final_array, mode="RGB")

        return final_image

    def find_faces(self, image_path: str,age_gender_detect) -> Dict:
        c_annotated_image_width = 400
        photogift_base_names = [
            # 'stp-Sock1','stp-Sock2','stp-Sock3','stp-Sock4','stp-Sock5','stp-Advent','stp-Cushion','stp-Keychain','stp-Mug','stp-Ornament',
            'WP-ornament', 'WP-cushion', 'WP-mug', 'WP-sock1', 'WP-sock2', 'WP-sock3', 'Christmasornament', 'keychain1', 'minileinwand', 'Bierkrug',
            'Fotocard', 'Fobofridge', 'Acrylickeychain']
        print("DEBUG:find_faces_started")
        base_dir = os.path.dirname(image_path)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        self.working_directory = os.path.join(base_dir, base_name)
        os.makedirs(self.working_directory, exist_ok=True)

        copied_image_name = os.path.basename(image_path)
        copied_image_path = os.path.join(self.working_directory, copied_image_name)
        shutil.copy2(image_path, copied_image_path)

        default_image_filename = f"{c_default_customerimage_name}.png"
        default_image_path = os.path.join(self.working_directory, default_image_filename)
        shutil.copy2(image_path, default_image_path)

        # freistellen mit remove_background
        default_image_filename_background_removed = f"{c_default_customerimage_name_no_Background}.png"
        default_image_path_removed_background = os.path.join(self.working_directory, default_image_filename_background_removed)
        remove_background(default_image_path, default_image_path_removed_background, background_rgb=(-1,-1,-1), backgroundimage=None)

        # Version: ganzes Bild , nicht freigestellt, gestretched, geschärft, clahe
        tempimg= r"D:\PythonBilder\Test\tempimg.png"
        temp_stretched_complete_picture = r"D:\PythonBilder\Test\tempstretched_complete_picture.png"
        temp_unsharp_complete_picture = r"D:\PythonBilder\Test\tempunsharp_complete_picture.png"
        grayscale_image_path_complete_picture = r"D:\PythonBilder\Test\grayscale_complete_picture.png"
        clahe_complete_picture = r"D:\PythonBilder\Test\clahe_complete_picture.png"
        clahe_masked_black_background_path = r"D:\PythonBilder\Test\clahe_masked_black_background.png"
        default_image_filename_grc_nobgd = f"{c_default_customerimage_name_grc_nobgd}.png"

        convert_to_grayscale(default_image_path, grayscale_image_path_complete_picture)
        stretch_grayscale(grayscale_image_path_complete_picture,temp_stretched_complete_picture , False)
        unsharp_mask(temp_stretched_complete_picture, temp_unsharp_complete_picture, scalesize=default_scalesize, amount=1.5, blur_radius=3)
        clahe_contrast(temp_stretched_complete_picture, clahe_complete_picture, clipLimit=3.0, tileGridSize=8, show_result=False)

        clahe_masked_black_background = self.apply_mask_and_black_background(clahe_path=clahe_complete_picture,mask_path=default_image_path_removed_background)
        clahe_masked_black_background.save(clahe_masked_black_background_path)

        # Graustufe Stretch und unsharp masking
        temp_stretched = r"D:\PythonBilder\Test\tempstretched.png"
        temp_unsharp = r"D:\PythonBilder\Test\tempunsharp.png"
        grayscale_image_path = r"D:\PythonBilder\Test\grayscale.png"
        convert_to_grayscale(default_image_path_removed_background, grayscale_image_path)
        stretch_grayscale(default_image_path_removed_background,temp_stretched , False)
        unsharp_mask(temp_stretched, temp_unsharp, scalesize=default_scalesize, amount=1.5, blur_radius=3)

        # Clahe Kontrast Filter mit Hintergrund
        default_image_filename_grc = f"{c_default_customerimage_name_grc}.png"
        default_image_path_grc = os.path.join(self.working_directory, default_image_filename_grc)
        clahe_contrast(default_image_path, default_image_path_grc, clipLimit=3.0, tileGridSize=8, show_result=False)

        image = Image.open(copied_image_path).convert("RGB")   # das normale Bild wird geladen
        image_np = np.array(image)
        faces = self.get_face_app().get(image_np)

        image_no_background = Image.open(default_image_path_removed_background).convert("RGBA") # das freigestellte Bild wird geladen

        orig_width, orig_height = image.size
        scale_ratio = c_annotated_image_width / orig_width
        scaled_height = int(orig_height * scale_ratio)
        image_scaled = image.resize((c_annotated_image_width, scaled_height), Image.LANCZOS)
        draw = ImageDraw.Draw(image_scaled)

        try: font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=12)
        except IOError: font = ImageFont.load_default()
        results = []

        # These belong in the top-level metadata, not in the faces list
        original_image_name = copied_image_name
        original_image_path = default_image_path

        for i, face in enumerate(faces, start=1):
            box = face.bbox.astype(int)
            x1, y1, x2, y2 = [int(coord) for coord in box]
            width = x2 - x1
            height = y2 - y1

            margin_ratio = 0.3
            margin_x = int(width * margin_ratio)
            margin_y = int(height * margin_ratio)

            new_x1 = max(x1 - margin_x, 0)
            new_y1 = max(y1 - margin_y, 0)
            new_x2 = min(x2 + margin_x, image.width)
            new_y2 = min(y2 + margin_y, image.height)

            # erst die Ausschnitte ohne Hintergrund Entfernung
            crop_with_bg = image.crop((new_x1, new_y1, new_x2, new_y2))
            fname = f"{c_default_customerimage_name}_face_{i}.jpg"
            fpath = os.path.join(self.working_directory, fname)
            crop_with_bg.save(fpath)

            # und nun "ausschneiden" aus dem "Hintergrund entferntem" Bild
            crop_nobg = image_no_background.crop((new_x1, new_y1, new_x2, new_y2))
            filename_crop_nobg = f"{c_default_customerimage_name}_face_nobg_{i}.png"
            file_path_crop_nobg = os.path.join(self.working_directory, filename_crop_nobg)
            crop_nobg.save(file_path_crop_nobg)

            #  "ausschneiden" aus dem clahe kontrastiertem Bild
            crop_clahe = clahe_masked_black_background.crop((new_x1, new_y1, new_x2, new_y2))
            fname2 = f"{c_default_customerimage_name}_face_black_nobg_clahe{i}.png"
            fpath2 = os.path.join(self.working_directory, fname2)
            crop_clahe.save(fpath2)

            # erzeuge die Fotogeschenke Bilder

            for base_name in photogift_base_names:
                filename_photogift = f"{c_default_customerimage_name}_face_{i}_{base_name}.png"
                filepath_photogift = os.path.join(self.working_directory, filename_photogift)
                photogift = photogift_composer()
                #result = photogift.apply(file_path_crop_nobg, filepath_photogift)
                result = photogift.apply(base_name,fpath, filepath_photogift)

            # Jetzt noch das starterpackage Fotogeschenke Bild im Blister
            filename_starterpackage = f"{c_default_customerimage_name}_face_{i}_starterpackage.png"
            filepath_starterpackage = os.path.join(self.working_directory, filename_starterpackage)
            photogift = photogift_composer()
            scenic_list = ['WP-ornament','WP-cushion','WP-mug','WP-sock1','WP-sock2','WP-sock3']
            fusioniertesBild = photogift.fusion(scenic_list, filepath_starterpackage)

            # Use FairFace for  age group and gender
            if age_gender_detect:
                age_group, age_conf, gender = self. estimate_age_and_gender(crop)
            else: print("Info: Erkennung von Alter und Geschlecht wurde deaktiviert und wird ausgelassen.")

            x1_scaled = int(x1 * scale_ratio)
            y1_scaled = int(y1 * scale_ratio)
            x2_scaled = int(x2 * scale_ratio)
            y2_scaled = int(y2 * scale_ratio)

            if not age_gender_detect: inner_color = "white"
            else: inner_color = "pink" if gender == "Female" else "green"
            draw.rectangle([x1_scaled - 1, y1_scaled - 1, x2_scaled + 1, y2_scaled + 1], outline="black", width=2)
            draw.rectangle([x1_scaled, y1_scaled, x2_scaled, y2_scaled], outline=inner_color, width=2)
            if age_gender_detect:
                label_text = f"{gender}, {age_group} ({age_conf:.2f})"
                draw.text((x1_scaled + 1, y1_scaled - 19), label_text, fill="black", font=font)
                draw.text((x1_scaled, y1_scaled - 20), label_text, fill=inner_color, font=font)

            if age_gender_detect:
                results.append({
                    #"image": copied_image_name,
                    "bounding_box": [x1, y1, x2, y2],
                    "age_group": age_group,
                    "age_confidence": age_conf,
                    "gender": gender,
                    "saved_file": fname
                })
            else:
                results.append({
                    "bounding_box": [x1, y1, x2, y2],
                    "saved_file": fname
                })

        ##############################
        animals = self.detect_animals(copied_image_path)
        ##############################
        #print("Debug: Tiere:", animals)

        z = 0
        for pet in animals:
            z = z +1
            x1, y1, x2, y2 = pet["bounding_box"]
            label = pet["label"].capitalize()
            conf = pet["confidence"]
            crop = image.crop((x1, y1, x2, y2))
            fname = f"{c_default_customerimage_name}_animal_{z}.jpg"
            #fname = f"{c_default_customerimage_name}_face_{i}.jpg"
            fpath = os.path.join(self.working_directory, fname)
            crop.save(fpath)

            for base_name in photogift_base_names:
                filename_photogift = f"{c_default_customerimage_name}_animal_{z}_{base_name}.png"
                filepath_photogift = os.path.join(self.working_directory, filename_photogift)
                photogift = photogift_composer()
                #result = photogift.apply(file_path_crop_nobg, filepath_photogift)
                print("Zeile 413 ",base_name,fpath, filepath_photogift)
                result = photogift.apply(base_name,fpath, filepath_photogift)

            # Jetzt noch das starterpackage Fotogeschenke Bild im Blister
            filename_starterpackage = f"{c_default_customerimage_name}_animal_{z}_starterpackage.png"
            filepath_starterpackage = os.path.join(self.working_directory, filename_starterpackage)
            photogift = photogift_composer()
            scenic_list = ['WP-ornament','WP-cushion','WP-mug','WP-sock1','WP-sock2','WP-sock3']
            fusioniertesBild = photogift.fusion(scenic_list, filepath_starterpackage)

            x1_scaled = int(x1 * scale_ratio)
            y1_scaled = int(y1 * scale_ratio)
            x2_scaled = int(x2 * scale_ratio)
            y2_scaled = int(y2 * scale_ratio)

            draw.rectangle([x1_scaled - 1, y1_scaled - 1, x2_scaled + 1, y2_scaled + 1], outline="blue", width=2)
            draw.rectangle([x1_scaled, y1_scaled, x2_scaled, y2_scaled], outline="cyan", width=2)
            draw.text((x1_scaled + 1, y1_scaled - 19), f"{label}, {conf:.2f}", fill="black", font=font)
            draw.text((x1_scaled, y1_scaled - 20), f"{label}, {conf:.2f}", fill="cyan", font=font)

        annotated_name = f"{base_name}_annotated.jpg"
        annotated_path = os.path.join(self.working_directory, annotated_name)
        image_scaled.save(annotated_path)

        metadata = {
            "original_image_name": original_image_name,
            "original_image_path": original_image_path,
            "image_size": tuple(image.size),
            "color_mode": image.mode,
            "number_of_faces": len(results),
            "number_of_animals": len(animals),
            "annotated_image": annotated_name,
            "faces": results,
            "animals": animals  # ← Add this
        }

        #print(json.dumps(metadata, indent=2))
        #print("[DEBUG] Metadata to be saved:", json.dumps(metadata, indent=2))

        return {
            "faces": results,
            "animals": animals,
            "annotated_image": annotated_name,
            "image_size": tuple(image.size),
            "color_mode": image.mode,
            "working_directory": self.working_directory,
            "original_image_name": original_image_name,
            "original_image_path": original_image_path
        }

    def detect_objects(self, image_path: str) -> List[str]:
        print("a")
        results = manager.get_yolo_model().predict(image_path)

        print("a")

#        results = self.yolo_model(image_path)
        detections = results[0].boxes.data.cpu().numpy()
        names = results[0].names
        print(f"[DEBUG] YOLO detected {len(detections)} objects in {image_path}")
        found_objects = set()
        for det in detections:
            x1, y1, x2, y2, conf, cls_id = det
            label = names[int(cls_id)]
            print(f"[DEBUG] Detected: {label} ({conf:.2f})")
            if label not in ["person", "dog", "cat"]:  # Skip if already handled
                found_objects.add(label.lower())
        return list(found_objects)

    def generate_prompts(self, faces: List[Dict], animals: List[Dict], image_path: str = "") -> List[str]:
        count = len(faces)
        age_groups = [f.get("age_group", "") for f in faces]
        genders = [f.get("gender", "") for f in faces]
        prompts = set()

        # Age group logic
        if count == 2 and len(age_groups) == 2:
            group_1, group_2 = age_groups
            if group_1 == group_2 and group_1 in ["20-29", "30-39"]:
                prompts.add("a romantic couple moment for a gift")
                prompts.add("a couple in love")
            elif {"0-2", "3-9", "10-19"}.intersection({group_1, group_2}) and {"30-39", "40-49", "50-59", "60-69", "70+"}.intersection(
                    {group_1, group_2}):
                prompts.add("a parent and child bonding photo")

        if count >= 3 and all(g not in ["0-2", "3-9", "10-19"] for g in age_groups):
            prompts.add("a group of adult friends sharing a memory")
            prompts.add("a friendship snapshot for a mug")

        if any(g in ["0-2", "3-9"] for g in age_groups):
            prompts.add("a family photo with children")
            prompts.add("a baby milestone memory")

        if any(g in ["60-69", "70+"] for g in age_groups):
            prompts.add("a tribute to a loved one")
            prompts.add("a generational family portrait")

        # Gender-based logic
        if all(g == "Female" for g in genders):
            prompts.add("a girls' day out photo")
        elif all(g == "Male" for g in genders):
            prompts.add("a boys' trip memory")

        # Filename-based hints
        filename = os.path.basename(image_path).lower()
        if "anniversary" in filename:
            prompts.add("a romantic anniversary photo")
        if "birthday" in filename:
            prompts.add("a birthday celebration photo")
        if "christmas" in filename:
            prompts.add("a Christmas greeting photo")
        if "valentine" in filename:
            prompts.add("a Valentine's Day card photo")
        if "wedding" in filename:
            prompts.add("a wedding moment for a canvas print")
        if "pet" in filename:
            prompts.add("a pet portrait for a cushion")

        # Pet-based logic
        if any(a.get("label") == "dog" for a in animals):
            prompts.add("a dog portrait for a cushion")
            prompts.add("a gift idea for dog lovers")
        if any(a.get("label") == "cat" for a in animals):
            prompts.add("a cat lover's gift photo")
            prompts.add("a cozy cat moment for a mug")

        # Always include fallback prompts
        prompts.update([
            "birthday party",
            "Christmas greeting",
            "Valentine's card",
            "vacation selfie",
            "thank-you gift photo",
            "a photo celebrating family love",
            "a vacation memory for a keepsake"
        ])

        return list(prompts)

    def smart_crop_and_pad(self,img, x1, y1, x2, y2, margin_ratio=0.3):
        if not isinstance(img, Image.Image):
            raise TypeError("Expected a PIL image, but got something else.")
        original_width, original_height = img.size

        # Expand bounding box with margin
        box_width = x2 - x1
        box_height = y2 - y1
        margin_x = int(box_width * margin_ratio)
        margin_y = int(box_height * margin_ratio)

        # Expanded box within image bounds
        x1_exp = max(x1 - margin_x, 0)
        y1_exp = max(y1 - margin_y, 0)
        x2_exp = min(x2 + margin_x, original_width)
        y2_exp = min(y2 + margin_y, original_height)

        # Missing pixels beyond image boundaries
        missing_left = max(0, margin_x - x1)
        missing_top = max(0, margin_y - y1)
        missing_right = max(0, (x2 + margin_x) - original_width)
        missing_bottom = max(0, (y2 + margin_y) - original_height)

        # Crop and convert to NumPy
        cropped = img.crop((x1_exp, y1_exp, x2_exp, y2_exp))
        cropped_np = np.array(cropped)

        # Final canvas size
        canvas_width = (x2_exp - x1_exp) + missing_left + missing_right
        canvas_height = (y2_exp - y1_exp) + missing_top + missing_bottom

        # Create canvas and paste cropped image
        canvas = Image.new("RGB", (canvas_width, canvas_height))
        canvas_np = np.array(canvas)

        # Paste cropped image at correct offset
        offset_x = missing_left
        offset_y = missing_top
        canvas_np[offset_y:offset_y + cropped_np.shape[0], offset_x:offset_x + cropped_np.shape[1], :] = cropped_np

        # Fill left
        if missing_left > 0:
            left_col = cropped_np[:, 0:1, :]
            left_fill = np.repeat(left_col, missing_left, axis=1)
            canvas_np[offset_y:offset_y + cropped_np.shape[0], 0:offset_x, :] = left_fill

        # Fill right
        if missing_right > 0:
            right_col = cropped_np[:, -1:, :]
            right_fill = np.repeat(right_col, missing_right, axis=1)
            canvas_np[offset_y:offset_y + cropped_np.shape[0], offset_x + cropped_np.shape[1]:, :] = right_fill

        # Fill top
        if missing_top > 0:
            top_row = canvas_np[offset_y:offset_y + 1, :, :]
            top_fill = np.repeat(top_row, missing_top, axis=0)
            canvas_np[0:offset_y, :, :] = top_fill

        # Fill bottom
        if missing_bottom > 0:
            bottom_row = canvas_np[offset_y + cropped_np.shape[0] - 1:offset_y + cropped_np.shape[0], :, :]
            bottom_fill = np.repeat(bottom_row, missing_bottom, axis=0)
            canvas_np[offset_y + cropped_np.shape[0]:, :, :] = bottom_fill

        return Image.fromarray(canvas_np)

    #def run(self, image_paths: List[str]) -> List[Dict]:
    def annotation_tool_run(self, image_paths: List[str],DetectAI,DetectObjects,DetectIntent) -> List[Dict]:
        results = []
        for path in image_paths:
            metadata = {}
            try: face_data = self.find_faces(path,DetectAI)
            except Exception as e:
                print(f"[ERROR77] Failed to process faces for {path}: {e}")
                continue  # Skip this image and move to the next

            metadata["faces"] = face_data["faces"]
            metadata["animals"] = face_data["animals"]
            animals = metadata["animals"]

            metadata["annotated_image"] = face_data["annotated_image"]
            metadata["image_size"] = face_data["image_size"]
            metadata["color_mode"] = face_data["color_mode"]
            metadata["original_image_name"] = face_data["original_image_name"]
            metadata["original_image_path"] = face_data["original_image_path"]

            # 🧠 Detect objects
            if DetectObjects: metadata["objects"] = list(set(label.lower() for label in self.detect_objects(path)))
            else: print("Info: Erkennung von Gegenständen wurde deaktiviert und wird ausgelassen.")

            # 📌 Add original image path
            metadata["image"] = path

            # 🧩 Load original image for face cropping
            try: o_img = Image.open(face_data["original_image_path"]).convert("RGB")
            except Exception as e:
                print(f"[ERROR] Loading original image: {e}")
                continue

            # 🧠 Couple detection and adaptive cropping
            faces = metadata["faces"]
            face_boxes = [np.array(face["bounding_box"]).astype(int) for face in faces]
            face_centers = [((box[0] + box[2]) / 2, (box[1] + box[3]) / 2) for box in face_boxes]
            couplet_exists = False
            c_trigger_closeness = 1.2
            for i, (face, box) in enumerate(zip(faces, face_boxes)):
                try:
                    x1, y1, x2, y2 = make_square_box(*box)
                    adaptive_margin = margin_ratio
                    current_center = face_centers[i]
                    box_width = abs(x2 - x1)
                    for j, other_center in enumerate(face_centers):
                        if i == j:
                            continue
                        dist = math.sqrt((current_center[0] - other_center[0]) ** 2 + (current_center[1] - other_center[1]) ** 2)
                        if dist < box_width * c_trigger_closeness:
                            adaptive_margin *= 0.01
                            if len(faces) == 2:
                                couplet_exists = True
                                # Get both face boxes
                                box1 = face_boxes[0]
                                box2 = face_boxes[1]

                                # Compute union of both boxes
                                x1_1, y1_1, x2_1, y2_1 = make_square_box(*box1)
                                x1_2, y1_2, x2_2, y2_2 = make_square_box(*box2)

                                couple_x1 = min(x1_1, x1_2)
                                couple_y1 = min(y1_1, y1_2)
                                couple_x2 = max(x2_1, x2_2)
                                couple_y2 = max(y2_1, y2_2)

                                #c_couple_adaptive_margin = margin_ratio  # You can adjust this if needed
                            break   # hier wird geprüft ob wir ein Paarbils haben
                    cropped = self.smart_crop_and_pad(o_img, x1, y1, x2, y2, adaptive_margin)
                    fname = f"{c_default_customerimage_name}_face_padded_{i+1}.jpg"
                    fpath = os.path.join(self.working_directory, fname)
                    cropped.save(fpath)
                    if couplet_exists:
                        # aus dem normalen Bild ausschneiden
                        couple_cropped = self.smart_crop_and_pad(o_img, couple_x1, couple_y1, couple_x2, couple_y2, c_couple_adaptive_margin)
                        couple_fname = f"{c_default_coupleimage_name}_couple_padded.jpg"
                        couple_fpath = os.path.join(self.working_directory, couple_fname)
                        couple_cropped.save(couple_fpath)
                        couple_metadata = {   # Daten für das json file
                            "couple_bounding_box": [int(x1), int(y1), int(x2), int(y2)],
                            "size": couple_cropped.size,
                            "margin_ratio": c_couple_adaptive_margin
                        }
                    cropped.thumbnail((150, 150), Image.Resampling.LANCZOS)
                except Exception as e: print(f"[ERROR] Processing face {i + 1}: {e}")
            metadata["couple"] = couplet_exists
            metadata["total_faces"] = len(faces)
            metadata["total_animals"] = len(animals)
            if couplet_exists:
                metadata["coupleimage"] = couple_fname
                metadata["couple_data"] = couple_metadata
            metadata_path = os.path.join(face_data["working_directory"], json_dateiname)  # 💾 Save metadata to JSON
            with open(metadata_path, "w") as f: json.dump(metadata, f, indent=2)
            results.append(metadata)
        return results

    def estimate_age_and_gender(self,face_image: Image.Image) -> Tuple[str, float, str]:
        age_results = self.get_age_classifier()(face_image)
        gender_results = self.get_gender_classifier()(face_image)

        # Get top age prediction
        age_group = next((r['label'] for r in age_results if '-' in r['label'] or '+' in r['label']), 'Unknown')
        age_conf = next((r['score'] for r in age_results if '-' in r['label'] or '+' in r['label']), 0.0)

        # Normalize gender prediction
        gender_raw = gender_results[0]['label'].lower()
        print(f"[DEBUG] Raw gender label: {gender_raw}")

        if gender_raw in ['male', 'man', 'boy']:
            gender = 'Male'
        elif gender_raw in ['female', 'woman', 'girl']:
            gender = 'Female'
        else:
            gender = 'Unknown'

        return age_group, round(age_conf, 2), gender

    def detect_animals(self, image_path: str) -> List[Dict]:
        print("1")

        results = manager.get_yolo_model().predict(image_path)

        print("2")

        detections = results[0].boxes.data.cpu().numpy()
        names = results[0].names
        image = Image.open(image_path)
        animals = []
        for i, det in enumerate(detections, start=1):
            x1, y1, x2, y2, conf, cls_id = det
            label = names[int(cls_id)]
            if label in ["dog", "cat"]:
                # Crop and save the animal image
                crop = image.crop((int(x1), int(y1), int(x2), int(y2)))
                fname = f"customerimage_animal_{i}.jpg"
                fpath = os.path.join(self.working_directory, fname)
                crop.save(fpath)
                animals.append({
                    "label": label,
                    "confidence": round(float(conf), 2),
                    "bounding_box": [int(x1), int(y1), int(x2), int(y2)],
                    "saved_file": fname
                })
        return animals

    #def remove_background_bria(self):
    #    remove_background(demo_image_path, output_image_path)

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


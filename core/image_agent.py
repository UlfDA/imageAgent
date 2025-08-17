"""""

image_toolkit/
│
├── core/
│   └── image_processor.py       # Core image processing class
│
├── gui/
│   └── tkinter_app.py           # Tkinter interface for testing
│
├── config/
│   └── init.ini                 # Stores selected image path
│
├── utils/
│   └── config_handler.py        # Handles reading/writing ini files
│
└── main.py                      # Entry point for GUI
"""

# core/image_agent.py

import os
import numpy as np
from typing import List, Dict, Tuple
from PIL import Image, ImageDraw, ImageFont
import insightface
from insightface.app import FaceAnalysis

class ImageAgent:
    def __init__(self,
                 model: str = "hog",
                 output_dir: str = "faces_output"):
        self.model = model
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def find_faces(
        self,
        image_path: str
    ) -> List[Dict[str, Tuple[int, int, int, int]]]:
        image       = face_recognition.load_image_file(image_path)
        bboxes      = face_recognition.face_locations(image, model=self.model)
        pil_img     = Image.fromarray(image)

        results = []
        for i, (top, right, bottom, left) in enumerate(bboxes, start=1):
            crop   = pil_img.crop((left, top, right, bottom))
            fname  = f"face_{i}.jpg"
            fpath  = os.path.join(self.output_dir, fname)
            crop.save(fpath)
            results.append({
                "bounding_box": (top, right, bottom, left),
                "saved_path":   fpath
            })
        return results

    def run(self, image_path: str):
        if not image_path:
            print("No image path provided.")
            return
        faces = self.find_faces(image_path)
        print(f"Detected {len(faces)} face(s) in {image_path}")
        for face in faces:
            print(" →", face["saved_path"])
# core/image_agent.py


class ImageAgent:
    def __init__(self,
                 model: str = "hog",
                 output_dir: str = "faces_output"):
        self.model = model
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        # Initialize InsightFace
        self.face_app = FaceAnalysis(providers=['CPUExecutionProvider'])  # Use GPU if available
        self.face_app.prepare(ctx_id=0, det_size=(640, 640))

    def find_faces(self, image_path: str) -> List[Dict[str, Tuple[int, int, int, int]]]:
        image = Image.open(image_path).convert("RGB")
        image_np = np.array(image)

        faces = self.face_app.get(image_np)
        draw = ImageDraw.Draw(image)

        results = []
        for i, face in enumerate(faces, start=1):
            box = face.bbox.astype(int)
            x1, y1, x2, y2 = box
            age = int(face.age)
            gender = "Male" if face.gender == 1 else "Female"

            # Crop and save face
            crop = image.crop((x1, y1, x2, y2))
            fname = f"face_{i}.jpg"
            fpath = os.path.join(self.output_dir, fname)
            crop.save(fpath)

            # Annotate original image
            label = f"{gender}, {age}"
            draw.rectangle([x1, y1, x2, y2], outline="green", width=2)
            draw.text((x1, y1 - 10), label, fill="blue")
            print({"bounding_box": (x1, y1, x2, y2),"age": age,"gender": gender, "saved_path": fpath })

            results.append({
                "bounding_box": (x1, y1, x2, y2),
                "age": age,
                "gender": gender,
                "saved_path": fpath
            })

        # Save annotated image
        annotated_path = os.path.join(self.output_dir, "annotated.jpg")
        image.save(annotated_path)
        print("Speicherpfad:",annotated_path)
        return results

    def run(self, image_path: str):
        if not image_path:
            print("No image path provided.")
            return
        faces = self.find_faces(image_path)
        print(f"Detected {len(faces)} face(s) in {image_path}")
        for face in faces:
            print(f" → {face['gender']} ({face['age']} yrs) saved at {face['saved_path']}")

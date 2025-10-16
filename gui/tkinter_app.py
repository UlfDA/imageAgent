#######################################
### Tkinter_app.py  Version greece
#######################################

json_dateiname = "MetadataAI.json"
import time
from tkinter import ttk

start = time.time()
print("[Startup] Beginning initialization...")
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from gui.image_creator import ImageCreator
from core.image_agent import ImageAgent
from utils.config_handler import ConfigHandler
import os
from .image_creator import ImageCreator

print(f"[time import]: loaded in {time.time() - start:.2f}s")

class ImageAnalyzeApp:
    def __init__(self, root, image_display_size=(400, 400)):
        self.root = root
        self.root.title("Image Toolkit")
        self.root.state("zoomed")  # Windows only
        self.root.geometry("1200x600")  # Wider layout for side-by-side frames
        self.original_photo = None
        self.annotated_photo = None
        self.currently_showing = "original"
        self.image_display_size = image_display_size
        self.agent = ImageAgent()
        self.config_handler = ConfigHandler()
        self.selected_paths = []
        self.thumbnail = None
        self.annotated_photo = None

        # Horizontal container
        self.main_frame = tk.Frame(root)
        self.main_frame.pack(pady=10, fill="both", expand=True)

        # Controls frame (left side)
        self.controls_frame = tk.LabelFrame(self.main_frame, text="Analyse Panel", padx=10, pady=10)
        self.controls_frame.pack(side="left", fill="y", padx=10)

        self.select_button = tk.Button(self.controls_frame, text="Auswahl: Einzelbild ", command=self.select_images)
        self.select_button.pack(pady=5)

        self.directory_button = tk.Button(self.controls_frame, text="Auswahl: Verzeichnis", command=self.process_directory_analysis)
        self.directory_button.pack(pady=5)

        self.go_button = tk.Button(self.controls_frame, text="Analysiere und generiere", command=self.go)
        self.go_button.pack(pady=5)

        self.exit_button = tk.Button(self.controls_frame, text="Exit", command=self.exit_app)
        self.exit_button.pack(pady=5)

        self.image_label = tk.Label(self.controls_frame, text="No images selected", wraplength=180, justify="left")
        self.image_label.pack(pady=5)

        # Image display frame (center-right)
        #self.image_group_frame = tk.LabelFrame(self.main_frame, text="Image Preview", padx=10, pady=10)
        #self.image_group_frame.pack(side="left", fill="both", expand=True, padx=10)

        self.face_thumbnails = []  # Prevent garbage collection
        self.animal_thumbnails = []

        self.original_text_label = tk.Label(self.controls_frame, text="Original", font=("Arial", 10, "italic"))
        self.original_text_label.pack()

        self.preview_label = tk.Label(self.controls_frame, text="No preview")
        self.preview_label.pack(pady=(0, 10))

        # Key press tracking
        self.key_pressed = tk.BooleanVar(value=False)
        self.root.bind("<Key>", self.on_key_press)

        # Load previously saved path
        saved_path = self.config_handler.get_image_path()
        if saved_path:
            self.selected_paths = [saved_path]
            self.image_label.config(text=f"Selected:\n{saved_path}")
            self.update_preview(saved_path)
        print(f"[Startup] debug9: loaded in {time.time() - start:.2f}s")

    def toggle_image(self):
        if self.currently_showing == "original":
            if self.annotated_photo:
                self.preview_label.config(image=self.annotated_photo)
                self.preview_label.image = self.annotated_photo
                #self.toggle_button.config(text="Show Original")
                self.currently_showing = "annotated"
        else:
            if self.original_photo:
                self.preview_label.config(image=self.original_photo)
                self.preview_label.image = self.original_photo
                #self.toggle_button.config(text="Show Annotated")
                self.currently_showing = "original"


    def on_key_press(self, event):
        print(f"Key pressed: {event.keysym}")
        self.key_pressed.set(True)

    def select_images(self):
        file_paths = filedialog.askopenfilenames(filetypes=[("Image Files", "*.png *.jpg *.jpeg")])
        if file_paths:
            self.selected_paths = list(file_paths)
            self.config_handler.save_image_path(self.selected_paths[0])
            display_text = "Selected:\n" + "\n".join(self.selected_paths)
            self.image_label.config(text=display_text)
            self.update_preview(self.selected_paths[-1])

    def update_preview(self, image_path):
        try:
            img = Image.open(image_path)
            target_width = self.image_display_size[0]
            aspect_ratio = img.height / img.width
            target_height = int(target_width * aspect_ratio)
            resized_img = img.resize((target_width, target_height), Image.LANCZOS)
            self.thumbnail = ImageTk.PhotoImage(resized_img)
            self.preview_label.config(image=self.thumbnail, text="")

            self.original_photo = ImageTk.PhotoImage(resized_img)
            self.preview_label.config(image=self.original_photo, text="")
            self.preview_label.image = self.original_photo

        except Exception as e:
            print(f"Error loading preview: {e}")
            self.preview_label.config(text="Preview failed", image="")

    def go(self):
        if not self.selected_paths:
            messagebox.showwarning("No Images", "Please select one or more images first.")
            return
        print("Processing images...:",self.selected_paths)
        results = self.agent.annotation_tool_run(self.selected_paths,False,False)
        if not results:
            messagebox.showinfo("No Faces", "No faces detected in the selected images.")
            tk.Label(self.results_label, text="No faces detected.").pack(anchor="w")
            return
        all_objects = set()  # Collect all unique objects across results
        for result in results:
            intent = result.get("intent", "Unknown")
            faces = result.get("faces", [])
            animals = result.get("animals", [])
            objects = result.get("objects", [])
            all_objects.update(objects)
        self.selected_image_path = self.selected_paths[-1]
        base_name = os.path.basename(self.selected_image_path)
        name_without_ext = os.path.splitext(base_name)[0]
        self.annotated_image_path = os.path.join(self.agent.working_directory, f"{name_without_ext}_annotated.jpg")
        self.show_results_on_main()
        if self.annotated_photo: self.toggle_image()

    def display_result(self, metadata):
        result_frame = tk.Frame(self.results_label, pady=10)
        result_frame.pack(fill="x", anchor="w")

        # Thumbnail
        thumb = Image.open(metadata["thumbnail_path"])
        thumb = thumb.resize((100, 100))
        thumb_img = ImageTk.PhotoImage(thumb)
        self.face_thumbnails.append(thumb_img)  # Prevent garbage collection

        thumb_label = tk.Label(result_frame, image=thumb_img)
        thumb_label.pack()

        # Line break
        tk.Label(result_frame, text="").pack()

        # Age, Gender, Intent
        age = metadata.get("age", "Unknown")
        gender = metadata.get("gender", "Unknown")
        intent = metadata.get("intent", "Unknown")
        info_text = f"Age: {age} | Gender: {gender} | Intent: {intent}"
        tk.Label(result_frame, text=info_text, font=("Arial", 10)).pack()

        # Objects
        objects = metadata.get("objects", [])
        object_text = ", ".join(objects) if objects else "No objects found"
        tk.Label(result_frame, text=f"Objects: {object_text}", wraplength=400, justify="left").pack()

    def show_results_on_main(self):
        if not self.agent or not self.agent.working_directory:
            print("No working directory set.")
            return

        base_name = os.path.basename(self.selected_image_path)
        name_without_ext = os.path.splitext(base_name)[0]
        annotated_path = os.path.join(self.agent.working_directory, f"{name_without_ext}_annotated.jpg")

        if not os.path.exists(annotated_path):
            print(f"Annotated image not found: {annotated_path}")
            return

        try:
            annotated_image = Image.open(annotated_path)
            target_width = self.image_display_size[0]
            aspect_ratio = annotated_image.height / annotated_image.width
            target_height = int(target_width * aspect_ratio)
            resized_image = annotated_image.resize((target_width, target_height), Image.LANCZOS)
            self.annotated_photo = ImageTk.PhotoImage(resized_image)
            #self.result_canvas.config(width=target_width, height=target_height)
            #self.result_canvas.delete("all")
            #self.result_canvas.create_image(target_width // 2, target_height // 2, image=self.annotated_photo)
            self.annotated_photo = ImageTk.PhotoImage(resized_image)

        except Exception as e:
            print(f"Error loading annotated image: {e}")

    def process_directory_analysis(self):

        start_time = time.time()  # ⏱️ Start timer
        counter = 0  # 📊 Initialize counter
        directory = filedialog.askdirectory(title="Select Directory")
        if not directory: return

        image_files = [
            os.path.join(directory, f)
            for f in os.listdir(directory)
            if f.lower().endswith((".png", ".jpg"))
        ]

        if not image_files:
            messagebox.showinfo("No Images", "No .png or .jpg files found in the selected directory.")
            return

        for image_path in image_files:
            counter += 1
            self.selected_paths = [image_path]
            self.selected_image_path = image_path
            self.image_label.config(text=f"Processing:\n{image_path}")
            self.update_preview(image_path)
            self.go()
            self.key_pressed.set(False)
            self.root.update()

        end_time = time.time()
        total_time = end_time - start_time
        average_time = total_time / counter if counter else 0

        # Format the message
        message = (
            "Verarbeitung abgeschlossen ✅\n\n"
            f"Insgesamt benötigte Zeit: {total_time:.2f} Sekunden\n"
            f"Durchschnittlich benötigte Zeit pro Bild: {average_time:.2f} Sekunden\n"
            f"Bearbeitete Bilder: {counter}"
        )

        # Create a temporary root window to show the messagebox
        root = tk.Tk()
        root.withdraw()  # Hide the root window
        messagebox.showinfo("Ergebnis", message)
        root.destroy()

    def exit_app(self):
        self.root.destroy()

#######################################
### Tkinter_image_creator  Version 1.10ter 25
#######################################

from image_tools import photogift_composer
import image_tools
print(image_tools.__file__)

import time
startCreator = time.time()        # zum debuggen der Laufzeiten der Classifikatoren; Auch die Ladezeiten sind beträchtlich; kann aber alles auch gelöscht werden
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from utils.config_handler import ConfigHandler
import os
import re
from .image_creator import ImageCreator
print(f"Creator 2: {time.time() - startCreator:.2f} seconds")
from core.image_agent import ImageAgent     # braucht 11 s
print(f"Creator 3: {time.time() - startCreator:.2f} seconds")
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
print(f"creator imports: {time.time() - startCreator:.2f} seconds")
json_dateiname = "MetadataAI.json"
default_photogift_thumbnail_size = 170
default_scalesize = 700

def cv2_to_pil(cv_img):
    # Convert BGR (OpenCV) to RGB
    rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb_img)

def overlay_by_brightness(customer_resized, negative_multiply_resized):
    # Convert both to float32
    customer = customer_resized.astype(np.float32) / 255.0
    overlay = negative_multiply_resized.astype(np.float32) / 255.0
    # Extract brightness from overlay
    brightness = cv2.cvtColor(overlay, cv2.COLOR_BGR2GRAY)  # shape: (H, W)
    brightness = np.clip(brightness, 0.0, 1.0)
    # Expand brightness to 3 channels
    alpha = cv2.merge([brightness, brightness, brightness])  # shape: (H, W, 3)
    result = overlay * alpha + customer * (1 - alpha)
    result = np.clip(result * 255, 0, 255).astype(np.uint8)
    return result

def multiply_blend(bottom_img, top_img):# multiply in photoshop
    bottom = np.asarray(bottom_img).astype(np.float32) / 255
    top = np.asarray(top_img).astype(np.float32) / 255
    result = bottom * top    # Multiplizieren-Modus anwenden
    result_img = Image.fromarray((result * 255).astype(np.uint8))       # Skaliere zurück zu [0, 255] und konvertiere zu uint8
    return result_img

def screen_blend(bottom_img, top_img):      # negativ multiplizieren in Photoshop
    bottom = bottom_img.astype(np.float32) / 255.0      # Ensure both images are float32 and normalized to [0, 1]
    top = top_img.astype(np.float32) / 255.0
    result = 1 - (1 - bottom) * (1 - top)       # Apply screen blend formula
    result = np.clip(result * 255, 0, 255).astype(np.uint8)     # Convert back to uint8
    return result

def blend_by_brightness(customer_resized, negative_multiply_resized):
    customer = customer_resized.astype(np.float32) / 255.0
    negative = negative_multiply_resized.astype(np.float32) / 255.0
    brightness = cv2.cvtColor(negative, cv2.COLOR_BGR2GRAY)  # shape: (H, W)
    brightness_3ch = cv2.merge([brightness, brightness, brightness])  # shape: (H, W, 3)
    result = customer * brightness_3ch
    result = np.clip(result * 255, 0, 255).astype(np.uint8)
    return result

def resize_with_aspect_ratio(image, scale_size):
    h, w = image.shape[:2]
    scale = scale_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
#########################################################################
### Classes
##################################################################
class ImageCreatorApp:
    def __init__(self, root, image_display_size=(400, 400)):
        self.root = root
        # Track image placement
        self.image_row = 0
        self.image_col = 0
        self.max_rows_per_column = 4  # Adjust based on screen height or preference

        self.root.title("Image Creator")
        self.root.state("zoomed")  # Windows only
        self.root.geometry("1200x600")  # Wider layout for side-by-side frames

        self.original_photo = None
        self.annotated_photo = None
        self.currently_showing = "original"

        self.image_display_size = image_display_size
        self.agent = ImageAgent()   #$
        self.config_handler = ConfigHandler()
        self.selected_paths = []
        self.thumbnail = None
        self.annotated_photo = None

        # Horizontal container
        self.main_frame = tk.Frame(root)
        self.main_frame.pack(pady=10, fill="both", expand=True)

        # Controls frame (left side)
        self.controls_frame = tk.LabelFrame(self.main_frame, text="Controls", padx=10, pady=10)
        self.controls_frame.pack(side="left", fill="y", padx=10)

        self.couple_status_label = tk.Label(self.controls_frame, text="", fg="blue")
        self.couple_status_label.pack(pady=5)

        self.select_button = tk.Button(self.controls_frame, text="Select Images", command=self.select_images)
        self.select_button.pack(pady=5)


        self.photogift_directory_button = tk.Button(self.controls_frame, text="Analysiere alle Bilder im Verzeichnis", command=self.process_directory_analysis)
        self.photogift_directory_button.pack(pady=5)

        self.go_button = tk.Button(self.controls_frame, text="Analysiere und generiere aktuelles Bild",command=lambda: self.analysiere_und_erzeuge_Fotogeschenke_single(self.ki_analyse_var.get()))
        self.go_button.pack(pady=5)

        self.exit_button = tk.Button(self.controls_frame, text="Exit", command=self.exit_app)
        self.exit_button.pack(pady=5)

        self.pause_until_keypressed = tk.BooleanVar(value=False)            # Boolean variable to hold Pause checkbox state

        self.pause_checkbox = tk.Checkbutton(           # Create the checkbox and link it to the variable
            self.controls_frame,
            text="Pausen im Ordner Modus",
            variable=self.pause_until_keypressed
        )
        self.pause_checkbox.pack(pady=5)

        # Boolean variable to hold checkbox state
        self.ki_analyse_var = tk.BooleanVar(value=False)

        # Create the checkbox and link it to the variable
        self.ki_checkbox = tk.Checkbutton( self.controls_frame,text="KI Analyse:",variable=self.ki_analyse_var)
        self.ki_checkbox.pack(pady=5)

        self.image_label = tk.Label(self.controls_frame, text="No images selected", wraplength=180, justify="left")
        self.image_label.pack(pady=5)

        self.face_thumbnails = []  # Prevent garbage collection
        self.animal_thumbnails = []

        self.original_text_label = tk.Label(self.controls_frame, text="Original", font=("Arial", 10, "italic"))
        self.original_text_label.pack()

        self.preview_label = tk.Label(self.controls_frame, text="No preview")
        self.preview_label.pack(pady=(0, 10))

        # Results frame
        self.results_frame = tk.LabelFrame(self.main_frame, text="Results",  width=450, padx=10, pady=10)
        self.results_frame.pack_propagate(False)  # Allow resizing based on content
        self.results_frame.pack(side="left", fill="y", padx=10)

        self.results_label = tk.Frame(self.results_frame)
        self.results_label.pack(fill="y", expand=True)

        # creator  frame (right of results )
        self.creator_frame = tk.LabelFrame(self.main_frame, text="Creator", padx=10, pady=10)
        self.creator_frame.pack_propagate(True)
        self.creator_frame.pack(side="left", fill="y", padx=10)

        self.creator_label = tk.Frame(self.creator_frame)
        self.creator_label.pack(fill="both", expand=True)

        self.face_thumbnails = []  # Prevent garbage collection
        self.animal_thumbnails = []

        # Load previously saved path
        saved_path = self.config_handler.get_image_path()
        if saved_path:
            self.selected_paths = [saved_path]
            self.image_label.config(text=f"Selected:\n{saved_path}")
        # demomode das zuletzt gespeicherte Bild wir automatisch bearbeitet            #
        self.update_preview(saved_path)

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

    def go(self,ki_analyse):
        if not self.selected_paths:
            messagebox.showwarning("No Images", "Please select one or more images first.")
            return
        print("Processing images...:",self.selected_paths)
        # ki_analyse = metadata wie geschlecht, alter erfassen ; 2ter bool: Objekt Analyse, 3ter: Absicht analyse
        results = self.agent.annotation_tool_run(self.selected_paths,ki_analyse,ki_analyse,ki_analyse)
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
        self.process_single_file()
        if self.annotated_photo: self.toggle_image()

    def clear_results_label(self):
        for widget in self.results_label.winfo_children():
            widget.destroy()
        self.face_thumbnails.clear()

    def clear_creator_label(self):
        # Track image placement
        self.image_row = 0
        self.image_col = 0

        for widget in self.creator_label.winfo_children():
            widget.destroy()

    def select_images(self):
        file_paths = filedialog.askopenfilenames(filetypes=[("Image Files", "*.png *.jpg *.jpeg")])
        if file_paths:
            self.selected_paths = list(file_paths)
            self.config_handler.save_image_path(self.selected_paths[0])
            display_text = "Selected:\n" + "\n".join(self.selected_paths)
            self.image_label.config(text=display_text)
            # Automatisch starten des ANzeigens
            self.update_preview(self.selected_paths[-1])

    def update_preview(self, image_path):
        try:
            self.clear_results_label()
            self.clear_creator_label()
            self.face_thumbnails.clear()

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
            self.show_json_results()
        except Exception as e:
            print(f"Error loading preview: {e}")
            # nennt sich hier wohl anders-> self.preview_label.config(text="Preview failed", image="")

    def pad_to_square(self, img, size=150, color=(255, 255, 255)):
        # Calculate the new size preserving aspect ratio
        original_width, original_height = img.size
        ratio = size / max(original_width, original_height)
        new_width = int(original_width * ratio)
        new_height = int(original_height * ratio)
        img = img.resize((new_width, new_height), Image.LANCZOS)
        # Create a new square canvas
        new_img = Image.new("RGB", (size, size), color)
        # Center the resized image on the canvas
        offset_x = (size - new_width) // 2
        offset_y = (size - new_height) // 2
        new_img.paste(img, (offset_x, offset_y))
        return new_img

    def process_single_file(self):
        if not self.selected_paths:
            messagebox.showinfo("No Selection", "No image file is currently selected.")
            return
        selected_path = self.selected_paths[-1]
        self.prepare_image_ui(selected_path)
        json_path = self.get_json_path(selected_path)
        if not os.path.exists(json_path):
            self.handle_missing_metadata(selected_path)
            return
        results = self.process_image_to_photogift(json_path)
        self.display_results(results,json_path)

    def analysiere_und_erzeuge_Fotogeschenke_single(self,ki_analyse):
        if self.selected_paths:      # Der User hat also einen oder mehrere Bilder selektiert
            first_path = self.selected_paths[0]
            json_path = self.get_json_path(first_path)
            print(f"Creator vor (go): {time.time() - startCreator:.2f} seconds")
            self.go(ki_analyse) #  21 Sekunden
            print(f"Creator nach (go): {time.time() - startCreator:.2f} seconds")
            results = self.process_image_to_photogift(json_path)   # nur eine s
            self.display_results(results, json_path)

    def show_json_results(self):
        if not self.selected_paths:
            messagebox.showwarning("No Images", "Please select one or more images first.")
            return
        self.clear_results_label()
        selected_path = self.selected_paths[-1]
        base_dir = os.path.dirname(selected_path)
        base_name = os.path.splitext(os.path.basename(selected_path))[0]
        json_path = os.path.join(base_dir, base_name, json_dateiname)

        if not os.path.exists(json_path):           # ✅ Check if metadata exists before processing
            print(f"[SKIP] No metadata for {json_path}")
            return

        creator = ImageCreator(
            working_directory=os.path.join(base_dir, base_name),
            pad_to_square=self.pad_to_square
        )
        results = creator.display_all(json_path)
        self.display_results(results,json_path)

    def show_photogift_in_creator_frame(self, image_path,scenic_name, label_var_name):
        image_path = image_path.replace(".jpg", "")
        filepath_photogift = f"{image_path}_{scenic_name}.png"
        result = cv2.imread(filepath_photogift, cv2.IMREAD_UNCHANGED)

        scaled_result = resize_with_aspect_ratio(result, default_scalesize)
        rgb_image = cv2.cvtColor(scaled_result, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_image)
        #pil_image.save(save_path, format='JPEG')
        pil_image.thumbnail((default_photogift_thumbnail_size, default_photogift_thumbnail_size), Image.Resampling.LANCZOS)

        # Display in creator frame
        photo = ImageTk.PhotoImage(pil_image)
        label = tk.Label(self.creator_label, image=photo, borderwidth=2, relief="solid")
        label.image = photo  # Prevent garbage collection
        label.grid(row=self.image_row, column=self.image_col, padx=5, pady=5)
        self.image_row += 1
        if self.image_row >= self.max_rows_per_column:
            self.image_row = 0
            self.image_col += 1
        setattr(self, label_var_name, label)

    def display_results(self, results, json_path):
        functioncounter = 0
        self.clear_results_label()
        self.clear_creator_label()
        self.face_thumbnails.clear()
        base_dir = os.path.dirname(json_path)

        # Horizontal container
        horizontal_frame = tk.Frame(self.results_label)
        horizontal_frame.pack(fill=tk.BOTH, expand=True)

        # Bildname
        bildname = results.get("original_image_name", "Unknown")
        label = tk.Label(horizontal_frame, text=f"Bild Name: {bildname}", font=("Arial", 12, "bold"))
        label.pack(fill=tk.X, padx=5, pady=(5, 2))

        # Left column
        left_column = tk.Frame(horizontal_frame)
        left_column.pack(side=tk.LEFT, anchor="n", padx=10, pady=10)

        # Right column container with canvas + scrollbar
        right_column_container = tk.Frame(horizontal_frame)
        right_column_container.pack(side=tk.LEFT, fill=tk.Y, expand=True, padx=10, pady=10)
        canvas = tk.Canvas(right_column_container)
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar = tk.Scrollbar(right_column_container, orient="vertical", command=canvas.yview)
        scrollbar.pack(side=tk.RIGHT, fill="y")
        canvas.configure(yscrollcommand=scrollbar.set)

        # Right column inside canvas
        right_column = tk.Frame(canvas)
        canvas_window = canvas.create_window((0, 0), window=right_column, anchor="nw")

        def resize_canvas(event):
            canvas.itemconfig(canvas_window, width=event.width)

        canvas.bind("<Configure>", resize_canvas)
        right_column.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))

        # === LEFT COLUMN CONTENT ===
        if results.get("annotated_image"):
            image_path = os.path.join(base_dir, results["annotated_image"])
            photo = Image.open(image_path)
            photo.thumbnail((350, 500))
            annotated_photo = ImageTk.PhotoImage(photo)

            ### DIese drei Zeile reaktivieren wenn das getagte Bild angezeigt werden soll
            #annotated_label = tk.Label(left_column, image=annotated_photo, borderwidth=2, relief="solid")
            #annotated_label.image = annotated_photo
            #annotated_label.pack(pady=5)


        tk.Label(left_column, text=f"Kontext: {results.get('intent', 'Unknown')}", font=("Arial", 12), wraplength=250, justify="left").pack(
            anchor="w")
        tk.Label(left_column, text=f"Anzahl Gesichter: {results.get('total_faces', 'Unknown')}", font=("Arial", 12)).pack(anchor="w")
        tk.Label(left_column, text=f"Anzahl Tierw: {results.get('total_animals', 'Unknown')}", font=("Arial", 12)).pack(anchor="w")

        couplet_exists = results.get("couple", False)
        couple_text = "Paar Bild: Ja" if couplet_exists else "Paar Bild: Nein"
        tk.Label(left_column, text=couple_text, font=("Arial", 12), fg="black" if couplet_exists else "gray").pack(anchor="w")

        objects = results.get("objects", [])
        if objects:
            tk.Label(left_column, text="Gefundene Objekte:", font=("Arial", 12)).pack(anchor="w", pady=(15, 5))
            object_text = ", ".join(sorted(objects))
            tk.Label(left_column, text=object_text, wraplength=400, justify="left").pack(anchor="w")

        # === RIGHT COLUMN CONTENT ===
        if results.get("coupleimage"):
            image_path = os.path.join(base_dir, results["coupleimage"])
            photo = Image.open(image_path)
            photo.thumbnail((400, 300))
            couple_photo = ImageTk.PhotoImage(photo)

            couple_label = tk.Label(right_column, image=couple_photo, borderwidth=2, relief="solid")
            couple_label.image = couple_photo
            couple_label.pack(pady=5)

            tk.Label(right_column, text="Paar Bild", font=("Arial", 12)).pack()

        functioncounter = functioncounter +1
        print("display_results  counter:",functioncounter)

        # Gesichter anzeigen
        counter= 0
        for face_data in results.get("faces", []):
            counter = counter +1
            print("Face loop counter:",counter)
            face_frame = tk.Frame(right_column)
            face_frame.pack(fill=tk.X, expand=True, pady=5)

            image_row = tk.Frame(face_frame)
            image_row.pack(fill=tk.X, expand=True)

            # Original face
            image_path = os.path.join(base_dir, face_data["saved_file"])
            photo1img = Image.open(image_path)
            photo1img.thumbnail((150, 150))
            photo1 = ImageTk.PhotoImage(photo1img)
            label1 = tk.Label(image_row, image=photo1, borderwidth=2, relief="solid")
            label1.image = photo1
            label1.grid(row=0, column=0, padx=5)
            self.face_thumbnails.append(photo1)

            # Enhanced face
            padded_path = re.sub(r'_(\d+)(\.jpg)$', r'_padded_\1\2', image_path)
            photo2img = Image.open(padded_path)
            photo2img.thumbnail((150, 150))
            photo2 = ImageTk.PhotoImage(photo2img)
            label2 = tk.Label(image_row, image=photo2, borderwidth=2, relief="solid")
            label2.image = photo2
            ### Aktivieren wenn das original ausgeschnittene Bild angezeigt werden soll label2.grid(row=0, column=1, padx=5)
            ### self.face_thumbnails.append(photo2)

            # contrast optimized clahe face
            clahe_path = re.sub(r'_(\d+)(\.jpg)$', r'_black_nobg_clahe\1\2', image_path)
            clahe_path = clahe_path.replace(".jpg", ".png")

            print("image_path_clahe:",clahe_path)
            #aus image_agent :image_path_clahe = f"{c_default_customerimage_name}_face_black_nobg_clahe{i}.png"

            #self.show_photogift_in_creator_frame(image_path,'stp-Cushion', 'creator_bierkrug_label')
            #self.show_photogift_in_creator_frame(image_path,'stp-Advent', 'creator_bierkrug_label')
            self.show_photogift_in_creator_frame(image_path,'Acrylickeychain', 'creator_bierkrug_label')
            self.show_photogift_in_creator_frame(image_path,'Fobofridge', 'creator_bierkrug_label')
            self.show_photogift_in_creator_frame(image_path,'Fotocard', 'creator_bierkrug_label')
            self.show_photogift_in_creator_frame(image_path,'Bierkrug', 'creator_bierkrug_label')
            self.show_photogift_in_creator_frame(image_path,'Christmasornament', 'creator_ornament_label')
            self.show_photogift_in_creator_frame(image_path,'keychain1', 'creator_keychain1_label')
            self.show_photogift_in_creator_frame(image_path,'minileinwand', 'creator_minileinwand_label')

            # weitere Daten
            gender = face_data.get("gender", "Unknown")
            age_group = face_data.get("age_group", "Unknown")
            info_text = f"Age: {age_group}  | Sex: {gender}"
            #tk.Label(face_frame, text=info_text, font=("Arial", 10)).pack(pady=5)
            tk.Label(face_frame, text=info_text, font=("Arial", 10), anchor='w', justify='left').pack(pady=5, anchor='w', fill='x')

        # Tiere anzeigen
        for animal_data in results.get("animals", []):
            animal_frame = tk.Frame(right_column)
            animal_frame.pack(anchor="w", pady=5)

            image_row = tk.Frame(animal_frame)
            image_row.pack()

            image_path = os.path.join(base_dir, animal_data["saved_file"])
            photo1img = Image.open(image_path)
            photo1img.thumbnail((150, 150))
            photo1 = ImageTk.PhotoImage(photo1img)

            label1 = tk.Label(image_row, image=photo1, borderwidth=2, relief="solid")
            label1.image = photo1
            label1.pack()

            label = animal_data.get("label", "Unknown").capitalize()
            conf = animal_data.get("confidence", 0.0)
            labeltext = f"{label} ({conf:.2f})"

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

        # Create a persistent status window before the loop
        status_window = tk.Toplevel()
        status_window.title("Status")
        status_label = tk.Label(status_window, text="", font=("Arial", 12), justify="left")
        status_label.pack(padx=20, pady=20)

        for image_path in image_files:
            counter += 1
            self.selected_paths = [image_path]
            self.selected_image_path = image_path

            # Update status window with current image info
            status_label.config(text=f"Nummer: {counter}\n{image_path}")
            status_window.update()

            self.image_label.config(text=f"Processing:\n{image_path}")
            #$$$ self.update_preview(image_path)
            self.show_results_on_main()
            self.root.update_idletasks()

            #self.go(ki_analyse)
            self.go(self.ki_analyse_var)
            json_path = self.get_json_path(image_path)
            results = self.process_image_to_photogift(json_path)   # nur eine s
            #self.key_pressed.set(False)
            self.root.update()

        # After loop: show final message in the same window
        end_time = time.time()
        total_time = end_time - start_time
        average_time = total_time / counter if counter else 0

        final_message = (
            "Verarbeitung abgeschlossen ✅\n\n"
            f"Insgesamt benötigte Zeit: {total_time:.2f} Sekunden\n"
            f"Durchschnittlich benötigte Zeit pro Bild: {average_time:.2f} Sekunden\n"
            f"Bearbeitete Bilder: {counter}\n\n"
            "Drücke eine Taste, um das Fenster zu schließen."
        )

        status_label.config(text=final_message)

        # Wait for key press to close the window
        def close_on_key(event):
            status_window.destroy()

        status_window.bind("<Key>", close_on_key)
        status_window.focus_set()
        status_window.mainloop()

#    def process_directory_for_photogifts(self):
#        directory = filedialog.askdirectory(title="Select Directory")
#        if not directory: return##
#
#        image_files = [
#            os.path.join(directory, f)
#            for f in os.listdir(directory)
#            if f.lower().endswith((".png", ".jpg", ".jpeg"))
#        ]
#
#        if not image_files:
#            messagebox.showinfo("No Images", "No image files found in the selected directory.")
#            return

        # Create a persistent status window before the loop
 #       status_window = tk.Toplevel()
 #       status_window.title("Status")
 #       status_label = tk.Label(status_window, text="", font=("Arial", 12), justify="left")
 #       status_label.pack(padx=20, pady=20)

  #      counter = 0  # 📊 Initialize counter
  #      start_time = time.time()  # ⏱️ Start timer

  #      for image_path in image_files:
  #          counter += 1
  #          self.prepare_image_ui(image_path)
  #          json_path = self.get_json_path(image_path)
  #          if not os.path.exists(json_path):
  #              self.handle_missing_metadata(image_path)
  #              continue
            # Update status window with current image info
 #           status_label.config(text=f"Nummer: {counter}\n{image_path}")
 #           status_window.update()
            # wieso hier?  results = self.process_imageprocess_image_to_photogift(json_path)
 #           self.display_results(results,json_path)

            #self.key_pressed.set(False)
            #self.root.wait_variable(self.key_pressed)

        # After loop: show final message in the same window
  #      end_time = time.time()
  #      total_time = end_time - start_time
  #      average_time = total_time / counter if counter else 0

 #       final_message = (
 #           "Verarbeitung abgeschlossen ✅\n\n"
 #           f"Insgesamt benötigte Zeit: {total_time:.2f} Sekunden\n"
 #           f"Durchschnittlich benötigte Zeit pro Bild: {average_time:.2f} Sekunden\n"
 #           f"Bearbeitete Bilder: {counter}\n\n"
 #           "Drücke eine Taste, um das Fenster zu schließen."
 #       )

  #      status_label.config(text=final_message)

        # Wait for key press to close the window
   #     def close_on_key(event):
   #         status_window.destroy()

    #    status_window.bind("<Key>", close_on_key)
    #    status_window.focus_set()
    #    status_window.mainloop()

    # --- Helper Methods ---

    def prepare_image_ui(self, image_path):
        self.selected_paths = [image_path]
        self.selected_image_path = image_path
        self.image_label.config(text=f"Processing:\n{image_path}")
        self.update_preview(image_path)

        self.status_label = tk.Label(
            self.results_label,
            text=f"Processing {image_path}...",
            font=("Arial", 10, "italic")
        )
        self.status_label.pack(pady=5)

    def get_json_path(self, image_path):
        base_dir = os.path.dirname(image_path)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        return os.path.join(base_dir, base_name, "MetadataAI.json")

    def handle_missing_metadata(self, image_path):
        print(f"[SKIP] No metadata for {image_path}")
        self.status_label.config(text="Metadata not found. Skipping.")
        self.root.update_idletasks()
        time.sleep(1)
        self.status_label.destroy()

    def process_image_to_photogift(self, json_path):
        creator = ImageCreator(
            working_directory=os.path.dirname(json_path),
            pad_to_square=self.pad_to_square
        )
        results = creator.display_all(json_path)
        #self.status_label.destroy()
        return results

    def display_couple_result(self, result):
        couple_label = tk.Label(
            self.results_label,
            text=f"Couple Detection: {result}",
            font=("Arial", 10, "italic"),
            fg="darkgreen"
        )
        couple_label.pack(pady=2)

    def display_scaled_image(self, image_path):
        c_maxsize = 200  # Constant max size

        try:
            img = Image.open(image_path)
            original_width, original_height = img.size

            # Check if scaling is needed
            if original_width > c_maxsize or original_height > c_maxsize:
                # Calculate scale ratio
                scale_ratio = min(c_maxsize / original_width, c_maxsize / original_height)
                new_width = int(original_width * scale_ratio)
                new_height = int(original_height * scale_ratio)
                img = img.resize((new_width, new_height), Image.LANCZOS)

            # Convert to PhotoImage
            photo = ImageTk.PhotoImage(img)

            # Clear previous image if any
            for widget in self.creator_frame.winfo_children(): widget.destroy()

            # Display image
            label = tk.Label(self.creator_frame, image=photo)
            label.image = photo  # Prevent garbage collection
            label.pack()

        except Exception as e:
            print(f"Error displaying image: {e}")

    def exit_app(self):
        self.root.destroy()



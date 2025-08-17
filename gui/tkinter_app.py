# gui/tkinter_app.py

import tkinter as tk
from tkinter import filedialog
from core.image_agent import ImageAgent
from utils.config_handler import ConfigHandler


class ImageApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Image Toolkit")
        self.root.geometry("300x200")

        self.agent = None
        self.config_handler = ConfigHandler()

        # Label to show selected image
        self.image_label = tk.Label(root, text="No image selected")
        self.image_label.pack(pady=5)

        # Check for previously saved image path
        saved_path = self.config_handler.get_image_path()
        if saved_path:
            self.image_label.config(text=f"Selected: {saved_path}")

        # Buttons
        self.select_button = tk.Button(root, text="Select Image", command=self.select_image)
        self.select_button.pack(pady=10)

        self.go_button = tk.Button(root, text="Go", command=self.go)
        self.go_button.pack(pady=10)

        self.exit_button = tk.Button(root, text="Exit", command=self.exit_app)
        self.exit_button.pack(pady=10)


    def select_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png *.jpg *.jpeg")])
        if file_path:
            self.config_handler.save_image_path(file_path)
            print(f"Image selected: {file_path}")
            self.image_label.config(text=f"Selected: {file_path}")

    def go(self):
        image_path = self.config_handler.get_image_path()
        if image_path:
            self.agent = ImageAgent(image_path)  #$#$
            self.agent.find_faces(image_path)
        else:
            print("No image selected.")

    def exit_app(self):
        if self.agent:
            del self.agent
        self.root.destroy()

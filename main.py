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

# main.py

import tkinter as tk
from gui.tkinter_app import ImageApp
import os
from typing import List, Dict, Tuple
from PIL import Image

from core.image_agent import ImageAgent
from utils.config_handler import ConfigHandler

# main.py

import tkinter as tk
from gui.tkinter_app import ImageApp

if __name__ == "__main__":
    root = tk.Tk()
    app = ImageApp(root)
    root.mainloop()

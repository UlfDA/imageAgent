# main.py Greece

import tkinter as tk
#from gui.tkinter_app import ImageAnalyzeApp
from gui.Tkinter_image_creator import ImageCreatorApp

if __name__ == "__main__":
    root = tk.Tk()
    #analyze_app = ImageAnalyzeApp(root)
    creator_app = ImageCreatorApp(root)

    root.mainloop()

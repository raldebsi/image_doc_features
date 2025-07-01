import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk, ImageDraw
import face_recognition
import numpy as np
from pdf2image import convert_from_path
import cv2
from collections import defaultdict, Counter
import math


def pil_to_cv2(pil_image):
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def find_portrait_rect(img_cv2):
    hsv = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 220], dtype=np.uint8)
    upper_white = np.array([180, 20, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_white, upper_white)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask_cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    portrait_boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = h / w
        if w * h > 10000 and 0.75 < aspect_ratio < 1.5:
            portrait_boxes.append((x, y, x + w, y + h))

    return portrait_boxes


def face_inside(face_box, portrait_box):
    fx1, fy1, fx2, fy2 = face_box
    px1, py1, px2, py2 = portrait_box
    return px1 <= fx1 and py1 <= fy1 and px2 >= fx2 and py2 >= fy2


def crop_non_white_borders(image, threshold=25):
    np_img = np.array(image)
    mask = np.any(np_img < (255 - threshold), axis=-1)
    coords = np.argwhere(mask)
    if coords.size == 0:
        return image
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    cropped = image.crop((x0, y0, x1, y1))
    return cropped


def rgb_luminance(r, g, b):
    return 0.299 * r + 0.587 * g + 0.114 * b


def color_group_key(rgb, chroma_thresh=25, lum_thresh=0.4):
    r, g, b = rgb
    lum = rgb_luminance(r, g, b)

    return (int(r // chroma_thresh),
            int(g // chroma_thresh),
            int(b // chroma_thresh),
            int(lum // (lum_thresh * 255)))


def find_dominant_image_color(image, chroma_thresh=25, lum_thresh=0.4):
    print("[INFO] Calculating dominant color from full image...")
    np_img = np.array(image.convert("RGB"))
    h, w, _ = np_img.shape
    pixels = np_img.reshape(-1, 3)

    groups = defaultdict(list)

    for px in pixels:
        key = color_group_key(px, chroma_thresh, lum_thresh)
        groups[key].append(tuple(px))

    group_sizes = {k: len(v) for k, v in groups.items()}
    top_group = max(group_sizes.items(), key=lambda x: x[1])[0]
    selected_color = groups[top_group][0]
    print(f"[INFO] Dominant color group: {top_group}, RGB: {tuple(selected_color)}")
    return (*selected_color, 255)

def find_background_box_color(image, box, chroma_thresh=25, lum_thresh=0.4):
    print(f"[INFO] Sampling background color inside box {box}")
    np_img = np.array(image.convert("RGB"))
    x1, y1, x2, y2 = map(int, box)

    if x1 >= x2 or y1 >= y2:
        print("[WARN] Invalid box dimensions.")
        return (255, 255, 255, 255)

    pixels = np_img[y1:y2, x1:x2].reshape(-1, 3)

    if not len(pixels):
        print("[WARN] No pixels inside box.")
        return (255, 255, 255, 255)

    groups = defaultdict(list)
    for px in pixels:
        key = color_group_key(px, chroma_thresh, lum_thresh)
        groups[key].append(tuple(px))

    group_sizes = {k: len(v) for k, v in groups.items()}
    top_group = max(group_sizes.items(), key=lambda x: x[1])[0]
    selected_color = np.mean(groups[top_group], axis=0).astype(int)

    print(f"[INFO] Dominant box color group: {top_group}, RGB: {tuple(selected_color)}")
    return (*selected_color, 255)


def find_dominant_edge_color(image, box, chroma_thresh=25, lum_thresh=0.4):
    print(f"[INFO] Calculating dominant edge color for box {box}")
    np_img = np.array(image.convert("RGB"))
    x1, y1, x2, y2 = map(int, box)
    h, w = np_img.shape[:2]
    pad = 5

    pixels = []

    if y1 - pad >= 0:
        pixels.extend(np_img[y1 - pad:y1, x1:x2].reshape(-1, 3))
    if y2 + pad < h:
        pixels.extend(np_img[y2:y2 + pad, x1:x2].reshape(-1, 3))
    if x1 - pad >= 0:
        pixels.extend(np_img[y1:y2, x1 - pad:x1].reshape(-1, 3))
    if x2 + pad < w:
        pixels.extend(np_img[y1:y2, x2:x2 + pad].reshape(-1, 3))

    if not pixels:
        print("[WARN] No edge pixels found.")
        return (255, 255, 255, 255)

    groups = defaultdict(list)

    for px in pixels:
        key = color_group_key(px, chroma_thresh, lum_thresh)
        groups[key].append(tuple(px))

    group_sizes = {k: len(v) for k, v in groups.items()}
    top_group = max(group_sizes.items(), key=lambda x: x[1])[0]
    selected_color = np.mean(groups[top_group], axis=0).astype(int)
    print(f"[INFO] Dominant edge color group: {top_group}, RGB: {tuple(selected_color)}")
    return (*selected_color, 255)


class ForgeryApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Forgery Annotation Tool")

        self.page_index = 0
        self.pages = []
        self.layers = {}

        self.outline_var = tk.IntVar()
        self.mask_var = tk.IntVar()
        self.tamper_var = tk.IntVar()
        self.fill_mode = tk.StringVar(value="Edge Sampling")

        control_frame = tk.Frame(root)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        tk.Checkbutton(control_frame, text="Show Outlines", variable=self.outline_var, command=self.refresh).pack(side=tk.LEFT)
        tk.Checkbutton(control_frame, text="Show Mask", variable=self.mask_var, command=self.refresh).pack(side=tk.LEFT)
        tk.Checkbutton(control_frame, text="Show Tamper", variable=self.tamper_var, command=self.refresh).pack(side=tk.LEFT)
        tk.Button(control_frame, text="Open File", command=self.load_file).pack(side=tk.RIGHT)

        tk.Label(control_frame, text="Fill Mode:").pack(side=tk.LEFT)
        combo = ttk.Combobox(control_frame, textvariable=self.fill_mode,
                             values=["Edge Sampling", "Image Dominant Color"], state="readonly", width=20)
        combo.pack(side=tk.LEFT)
        combo.bind("<<ComboboxSelected>>", self.on_fill_mode_change)

        self.window_width = 800
        self.control_bar_height = 40  # approx height of control frame
        self.initial_height = 250

        self.canvas = tk.Canvas(root, width=self.window_width)
        self.canvas.pack(side=tk.BOTTOM)

        self.root.geometry(f"{self.window_width}x{self.initial_height}")


    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("PDF or Images", "*.pdf *.png *.jpg *.jpeg")])
        if not file_path:
            return

        self.pages.clear()
        self.layers.clear()

        if file_path.lower().endswith(".pdf"):
            self.pages = convert_from_path(file_path, dpi=200)
        else:
            self.pages = [Image.open(file_path).convert("RGB")]

        for i, page in enumerate(self.pages):
            cropped = crop_non_white_borders(page)
            faces = self.detect_faces(cropped)
            portraits = self.detect_portraits(cropped)
            outlines = self.generate_outline_boxes(cropped, faces, portraits)

            self.layers[i] = {
                'original': cropped,
                'outlines': outlines,
                'forge': self.generate_forge_layer(cropped, outlines),
                'mask': self.generate_mask_layer(cropped.size, outlines),
                'outline_img': self.generate_outline_layer(cropped.size, outlines)
            }

        self.page_index = 0
        self.refresh()

    def on_fill_mode_change(self, event=None):
        if self.page_index in self.layers:
            print(f"[INFO] Fill mode changed to: {self.fill_mode.get()}")
            image = self.layers[self.page_index]['original']
            boxes = self.layers[self.page_index]['outlines']
            self.layers[self.page_index]['forge'] = self.generate_forge_layer(image, boxes)
            self.refresh()

    def detect_faces(self, image):
        img_array = np.array(image.convert("RGB"))
        boxes = face_recognition.face_locations(img_array, model="hog")
        return [(left, top, right, bottom) for top, right, bottom, left in boxes]

    def detect_portraits(self, image):
        cv2_img = pil_to_cv2(image)
        return find_portrait_rect(cv2_img)

    def generate_outline_boxes(self, image, faces, portraits):
        boxes = [(f, "face") for f in faces]
        img_w, img_h = image.size
        margin = 1

        for p in portraits:
            px1, py1, px2, py2 = p

            # Skip if touching image edge
            if px1 <= margin or py1 <= margin or px2 >= img_w - margin or py2 >= img_h - margin:
                print(f"[INFO] Skipping edge-touching portrait box: {p}")
                continue

            if any(face_inside(f, p) for f in faces):
                boxes.append((p, "portrait"))

        return boxes



    def generate_outline_layer(self, size, boxes):
        img = Image.new("RGBA", size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        for box, label in boxes:
            draw.rectangle(box, outline='red' if label == 'face' else 'blue', width=3)
        return img

    def generate_mask_layer(self, size, boxes):
        img = Image.new("RGBA", size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        for box, _ in boxes:
            draw.rectangle(box, fill=(255, 255, 255, 255))
        return img

    def generate_forge_layer(self, image, boxes):
        img = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)

        if self.fill_mode.get() == "Image Dominant Color":
            fill_color = find_dominant_image_color(image)
            for box, _ in boxes:
                draw.rectangle(box, fill=fill_color)
        else:
            for box, _ in boxes:
                fill_color = find_background_box_color(image, box)
                draw.rectangle(box, fill=fill_color)

        return img

    def refresh(self):
        if not self.pages:
            return

        base = self.layers[self.page_index]['original'].copy().convert("RGBA")

        if self.tamper_var.get():
            base = Image.alpha_composite(base, self.layers[self.page_index]['forge'])

        if self.mask_var.get():
            base = Image.alpha_composite(base, self.layers[self.page_index]['mask'])

        if self.outline_var.get():
            base = Image.alpha_composite(base, self.layers[self.page_index]['outline_img'])

        # Resize image to match window width
        img_w, img_h = base.size
        new_w = self.window_width
        new_h = int(img_h * (new_w / img_w))  # preserve aspect ratio

        resized = base.resize((new_w, new_h), Image.Resampling.LANCZOS)

        # Resize canvas and window height accordingly
        self.tk_img = ImageTk.PhotoImage(resized)
        self.canvas.config(width=new_w, height=new_h)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)

        self.root.geometry(f"{self.window_width}x{new_h + self.control_bar_height}")



# === Run GUI ===
if __name__ == "__main__":
    root = tk.Tk()
    app = ForgeryApp(root)
    root.mainloop()

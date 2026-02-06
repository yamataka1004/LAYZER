import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog, ttk, scrolledtext
import sys
import cv2
import numpy as np
from PIL import Image, ImageTk
import csv
import os
import datetime

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

class GreenLeafApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Green Leaf Area Calculator")
        self.root.geometry("1200x800")

        # Application State
        self.cv_image = None        # Original OpenCV image (BGR)
        self.cv_image_small = None  # Resized image for display performance
        self.preview_scale = 1.0    # Scale of small image relative to original
        
        self.cached_display_vis = None # Cache for resizing
        self.resize_timer = None       # Timer for resize debounce
        
        self.display_image = None   # Image for Tkinter display
        self.original_h = 0
        self.original_w = 0
        
        # View transformations
        self.is_fit_mode = True     # If True, auto-fit to window
        self.zoom_scale = 1.0       # Current zoom (relative to cv_image_small)
        self.pan_offset_x = 0       # Pan offset X (canvas coords)
        self.pan_offset_y = 0       # Pan offset Y (canvas coords)
        self.drag_start_x = 0
        self.drag_start_y = 0
        
        self.pixels_per_cm = 0      # Calibration factor
        self.roi_points = []        # ROI Polygon points (scaled to original image)
        self.roi_mask = None        # Binary mask from ROI
        self.threshold_val = 0      # Current threshold value
        
        self.mode = "view"          # Modes: view, scale, roi, result
        self.temp_points = []       # Temporary points for drawing
        self.scale_line_points = [] # Stored points for scale line visualization

        self.file_list = []         # List of files to process
        self.current_file_index = -1
        self.unsaved_changes = False
        self.current_file_path = ""
        
        # Intercept close event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

        # Layout
        self.setup_ui()

    def setup_ui(self):
        # Tools Panel (Right Side)
        self.panel = tk.Frame(self.root, width=250, bg="#f0f0f0")
        self.panel.pack(side=tk.RIGHT, fill=tk.Y)
        self.panel.pack_propagate(False)

        # Padding for buttons
        pad_opts = {'padx': 10, 'pady': 5, 'fill': tk.X}

        tk.Label(self.panel, text="Controls", bg="#f0f0f0", font=("Arial", 12, "bold")).pack(pady=10)

        self.btn_load = tk.Button(self.panel, text="1. Open Files", command=self.open_files, bg="#e1e1e1")
        self.btn_load.pack(**pad_opts)
        
        # Navigation
        frame_nav = tk.Frame(self.panel, bg="#f0f0f0")
        frame_nav.pack(fill=tk.X, padx=5, pady=2)
        
        self.btn_prev = tk.Button(frame_nav, text="< Prev", command=lambda: self.navigate_image(-1), state=tk.DISABLED)
        self.btn_prev.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=2)
        
        self.btn_next = tk.Button(frame_nav, text="Next >", command=lambda: self.navigate_image(1), state=tk.DISABLED)
        self.btn_next.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=2)

        tk.Frame(self.panel, height=2, bg="#cccccc").pack(fill=tk.X, pady=10)

        # Help Button
        self.btn_help = tk.Button(self.panel, text="? How to Use", command=self.show_help, bg="#b3e5fc")
        self.btn_help.pack(**pad_opts)
        
        tk.Frame(self.panel, height=2, bg="#cccccc").pack(fill=tk.X, pady=10)

        self.btn_scale = tk.Button(self.panel, text="2. Set Scale", command=self.start_scale_mode, state=tk.DISABLED)
        self.btn_scale.pack(**pad_opts)
        self.lbl_scale = tk.Label(self.panel, text="Scale: Not Set", bg="#f0f0f0", fg="red")
        self.lbl_scale.pack(pady=2)

        tk.Frame(self.panel, height=2, bg="#cccccc").pack(fill=tk.X, pady=10)

        self.btn_roi = tk.Button(self.panel, text="3. Draw ROI (Polygon)", command=self.start_roi_mode, state=tk.DISABLED)
        self.btn_roi.pack(**pad_opts)
        self.btn_reset_roi = tk.Button(self.panel, text="Reset ROI", command=self.reset_roi, state=tk.DISABLED)
        self.btn_reset_roi.pack(**pad_opts)

        tk.Frame(self.panel, height=2, bg="#cccccc").pack(fill=tk.X, pady=10)

        tk.Label(self.panel, text="4. Threshold Adjustment", bg="#f0f0f0").pack(pady=5)
        
        # Threshold Controls Frame
        thresh_frame = tk.Frame(self.panel, bg="#f0f0f0")
        thresh_frame.pack(fill=tk.X, padx=5)

        # Value Entry and +/- Buttons
        val_frame = tk.Frame(thresh_frame, bg="#f0f0f0")
        val_frame.pack(fill=tk.X, pady=2)
        
        tk.Button(val_frame, text="-", width=3, command=lambda: self.increment_threshold(-0.1)).pack(side=tk.LEFT)
        self.val_threshold = tk.DoubleVar(value=0.0)
        self.ent_threshold = tk.Entry(val_frame, textvariable=self.val_threshold, width=8, justify='center')
        self.ent_threshold.pack(side=tk.LEFT, padx=5)
        # Verify input on enter
        self.ent_threshold.bind('<Return>', lambda e: self.on_threshold_result_entry())
        tk.Button(val_frame, text="+", width=3, command=lambda: self.increment_threshold(0.1)).pack(side=tk.LEFT)

        # Slider
        self.scale_threshold = tk.Scale(thresh_frame, from_=0, to=50, orient=tk.HORIZONTAL, variable=self.val_threshold, resolution=0.1, showvalue=0, command=self.on_threshold_change)
        self.scale_threshold.pack(fill=tk.X, pady=5)
        
        self.var_otsu = tk.BooleanVar(value=True)
        self.chk_otsu = tk.Checkbutton(self.panel, text="Auto (Otsu)", variable=self.var_otsu, command=self.toggle_otsu, bg="#f0f0f0")
        self.chk_otsu.pack(pady=2)

        tk.Frame(self.panel, height=2, bg="#cccccc").pack(fill=tk.X, pady=10)

        self.lbl_area = tk.Label(self.panel, text="Area: - cm²", font=("Arial", 14, "bold"), bg="#f0f0f0")
        self.lbl_area.pack(pady=10)

        self.btn_calc = tk.Button(self.panel, text="Calculate Area", command=self.calculate_area, state=tk.DISABLED, bg="#d1e7dd")
        self.btn_calc.pack(**pad_opts)

        self.btn_save = tk.Button(self.panel, text="Save Results", command=self.save_results, state=tk.DISABLED, bg="#4CAF50", fg="white")
        self.btn_save.pack(**pad_opts)
        
        # Canvas (Left Side)
        self.canvas_frame = tk.Frame(self.root, bg="gray")
        self.canvas_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(self.canvas_frame, bg="gray", cursor="cross")
        self.canvas.pack(fill=tk.BOTH, expand=True)

        # Events
        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<Motion>", self.on_canvas_move)
        # Right click to close polygon in ROI mode
        self.canvas.bind("<Button-3>", self.on_canvas_rclick) 
        
        # Resize event
        self.canvas.bind("<Configure>", self.on_canvas_configure)
        
        # Pan / Zoom Events
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)  # Windows
        self.canvas.bind("<Button-4>", self.on_mouse_wheel)    # Linux Scroll Up
        self.canvas.bind("<Button-5>", self.on_mouse_wheel)    # Linux Scroll Down
        
        self.canvas.bind("<ButtonPress-2>", self.on_drag_start) # Middle click (Wheel click)
        self.canvas.bind("<B2-Motion>", self.on_drag_move)

    def on_canvas_configure(self, event):
        if self.cv_image_small is None: return
        # Debounce resize
        if self.resize_timer is not None:
            self.root.after_cancel(self.resize_timer)
        self.resize_timer = self.root.after(50, self.process_resize)
        
    def process_resize(self):
        if self.cached_display_vis is not None:
             # Only re-fit if we are in fit mode
             if self.is_fit_mode:
                 self.show_image(self.cached_display_vis)
             self.resize_timer = None 

    def on_mouse_wheel(self, event):
        if self.cv_image_small is None: return
        
        # Determine zoom direction
        if event.num == 5 or event.delta < 0:
            scale_mult = 0.9
        else:
            scale_mult = 1.1
            
        # If in fit mode, initialize zoom state from current fit
        if self.is_fit_mode:
            self.is_fit_mode = False
            # Current zoom_scale is already set by show_image in fit mode
        
        # Calculate new scale
        new_scale = self.zoom_scale * scale_mult
        
        # Limit zoom
        if new_scale < 0.1: new_scale = 0.1
        if new_scale > 20.0: new_scale = 20.0
        
        # Zoom centered on mouse cursor
        # Mouse pos in canvas
        mx = self.canvas.canvasx(event.x)
        my = self.canvas.canvasy(event.y)
        
        # Image point under mouse (relative to top-left of image in canvas units, before zoom)
        # current_pos_x = pan_offset_x + img_x * zoom
        # img_x * zoom = mx - pan_offset_x
        
        # We want: new_pos_x = new_pan + img_x * new_zoom = mx
        # So: new_pan = mx - (img_x * new_zoom)
        #             = mx - ((mx - old_pan) / old_zoom) * new_zoom
        
        self.pan_offset_x = mx - ((mx - self.pan_offset_x) / self.zoom_scale) * new_scale
        self.pan_offset_y = my - ((my - self.pan_offset_y) / self.zoom_scale) * new_scale
        
        self.zoom_scale = new_scale
        self.update_display_transform()

    def on_drag_start(self, event):
        self.drag_start_x = event.x
        self.drag_start_y = event.y
        self.is_fit_mode = False # Dragging breaks fit mode

    def on_drag_move(self, event):
        dx = event.x - self.drag_start_x
        dy = event.y - self.drag_start_y
        
        self.pan_offset_x += dx
        self.pan_offset_y += dy
        
        self.drag_start_x = event.x
        self.drag_start_y = event.y
        
        self.update_display_transform()

    def open_files(self):
        # Support selecting multiple files
        file_paths = filedialog.askopenfilenames(
            title="Select Images",
            filetypes=[("Images", "*.jpg;*.jpeg;*.png;*.tif;*.tiff;*.bmp")]
        )
        if not file_paths:
            return

        self.file_list = list(file_paths)
        self.current_file_index = 0
        self.load_current_image()

    def navigate_image(self, step):
        if not self.file_list:
            return

        # Check unsaved changes
        if self.unsaved_changes:
            response = messagebox.askyesnocancel("Unsaved Changes", "You have not saved results for this image.\nSave now?")
            if response is True:
                self.save_results()
            elif response is None: # Cancel
                return
            # If False, discard and move on
            
        new_index = self.current_file_index + step
        if 0 <= new_index < len(self.file_list):
            self.current_file_index = new_index
            self.load_current_image()
        else:
            messagebox.showinfo("Navigation", "End of file list reached.")

    def load_current_image(self):
        if not self.file_list or self.current_file_index < 0:
            return
            
        file_path = self.file_list[self.current_file_index]
        
        try:
            self.current_file_path = file_path
            # Read image in BGR
            # cv2.imdecode for unicode path support usually better, but cv2.imread is ok for now if paths are standard
            # Using numpy to safety read if paths have weird chars
            with open(file_path, "rb") as f:
                bytes_arr = bytearray(f.read())
                numpy_arr = np.frombuffer(bytes_arr, dtype=np.uint8)
                img = cv2.imdecode(numpy_arr, cv2.IMREAD_COLOR)

            if img is None:
                raise ValueError("Could not read image file.")
            
            self.cv_image = img
            self.original_h, self.original_w = img.shape[:2]
            
            # Create small version for faster processing
            # Max dimension 1280 makes it fast enough for realtime updates while keeping detail
            max_dim = 1200
            if max(self.original_h, self.original_w) > max_dim:
                if self.original_h > self.original_w:
                    new_h = max_dim
                    new_w = int(self.original_w * (max_dim / self.original_h))
                else:
                    new_w = max_dim
                    new_h = int(self.original_h * (max_dim / self.original_w))
                self.cv_image_small = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
                self.preview_scale = new_w / self.original_w
            else:
                self.cv_image_small = img.copy()
                self.preview_scale = 1.0
            
            # Reset states for new image
            self.roi_points = []
            self.roi_mask = None
            # Keep pixels_per_cm if user wants to reuse scale across similar images? 
            # Plan didn't specify, but usually safer to reset or prompt? 
            # For this task, let's NOT reset pixels_per_cm automatically? 
            # Actually, standard behavior is usually reset unless "global scale" option. 
            # But here, let's reset to be safe, avoiding wrong measurements.
            self.pixels_per_cm = 0 
            self.lbl_scale.config(text="Scale: Not Set", fg="red")
            self.scale_line_points = [] # Reset scale line for new image
            self.lbl_area.config(text="Area: - cm²")
            self.temp_points = []
            self.mode = "view"
            self.unsaved_changes = False
            self.last_area_cm2 = None
            
            # Reset view to fit
            self.is_fit_mode = True
            
            # Update Title
            self.root.title(f"Green Leaf Area Calculator - [{self.current_file_index+1}/{len(self.file_list)}] {os.path.basename(file_path)}")
            
            # Enable buttons
            self.btn_scale.config(state=tk.NORMAL)
            self.btn_roi.config(state=tk.NORMAL)
            self.btn_reset_roi.config(state=tk.NORMAL)
            self.btn_calc.config(state=tk.NORMAL)
            self.btn_save.config(state=tk.NORMAL)
            
            # Update Nav Buttons
            self.btn_prev.config(state=tk.NORMAL if self.current_file_index > 0 else tk.DISABLED)
            self.btn_next.config(state=tk.NORMAL if self.current_file_index < len(self.file_list) - 1 else tk.DISABLED)
            
            # Show image (using small version)
            self.show_image(self.cv_image_small)
            
            # Calculate Otsu immediately for preview
            self.calculate_exg_otsu()

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {file_path}\n{e}")

    def on_closing(self):
        if self.unsaved_changes:
            if messagebox.askyesno("Quit", "You have unsaved changes. Really quit?"):
                self.root.destroy()
        else:
            self.root.destroy()

    def show_image(self, img_bgr):
        # Update cache (img_bgr is usually cv_image_small)
        self.cached_display_vis = img_bgr
        self.update_display_transform()

    def update_display_transform(self):
        if self.cached_display_vis is None: return

        # Get canvas dimensions
        canvas_w = self.canvas.winfo_width()
        canvas_h = self.canvas.winfo_height()
        if canvas_w < 10 or canvas_h < 10:
            canvas_w = 800
            canvas_h = 600

        img = self.cached_display_vis # This is small image
        h, w = img.shape[:2]

        if self.is_fit_mode:
             # Calculate scale to fit
             scale_w = canvas_w / w
             scale_h = canvas_h / h
             self.zoom_scale = min(scale_w, scale_h)
             
             # Center
             new_w = int(w * self.zoom_scale)
             new_h = int(h * self.zoom_scale)
             self.pan_offset_x = (canvas_w - new_w) // 2
             self.pan_offset_y = (canvas_h - new_h) // 2
        
        # Render
        # We need to resize the image to current zoom_scale
        # Note: zoom_scale is relative to cv_image_small
        
        # To avoid performance issues with huge zoom, we might want to crop?
        # For now, just resize entire small image. cv_image_small is max 1200px, so 2x zoom is 2400px.
        # 10x zoom is 12000px. That might be slow.
        # But usually users zoom in to see details.
        # Tkinter limit is high but not infinite.
        
        # Optimization: If zoom is very high, we could crop source, but that complicates rotation/etc logic (not used here).
        # Let's try simple resize first.
        
        # Actual display size
        disp_w = int(w * self.zoom_scale)
        disp_h = int(h * self.zoom_scale)
        
        if disp_w < 1 or disp_h < 1: return # Too small
        
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(self.cached_display_vis, cv2.COLOR_BGR2RGB)
        
        # Resize
        resized = cv2.resize(img_rgb, (disp_w, disp_h), interpolation=cv2.INTER_NEAREST if self.zoom_scale > 2 else cv2.INTER_AREA)
        
        self.pil_image = Image.fromarray(resized)
        self.display_image = ImageTk.PhotoImage(self.pil_image)
        
        self.canvas.delete("all")
        # Place at pan_offset
        self.canvas.create_image(self.pan_offset_x, self.pan_offset_y, anchor=tk.NW, image=self.display_image)
        
        # Redraw overlays
        self.redraw_overlays()

    def get_display_scale_factor(self):
        # Returns total scale factor from Original Image -> Canvas
        # original * preview_scale * zoom_scale = canvas
        if self.original_w == 0: return 1.0
        return self.preview_scale * self.zoom_scale

    def redraw_overlays(self):
        scale = self.get_display_scale_factor()
        ox, oy = self.pan_offset_x, self.pan_offset_y
        
        # ROI
        if len(self.roi_points) > 0:
            scaled_roi = []
            for p in self.roi_points:
                sx = int(p[0] * scale) + ox
                sy = int(p[1] * scale) + oy
                scaled_roi.append(sx)
                scaled_roi.append(sy)
            
            if len(self.roi_points) >= 2:
                self.canvas.create_polygon(scaled_roi, outline="yellow", fill="", width=2, tags="roi")
                # Highlight vertices
                for i in range(0, len(scaled_roi), 2):
                    px, py = scaled_roi[i], scaled_roi[i+1]
                    self.canvas.create_oval(px-3, py-3, px+3, py+3, fill="yellow", outline="red", tags="roi")

        # Persisted Scale Line
        if hasattr(self, 'scale_line_points') and len(self.scale_line_points) == 2:
            p1 = self.scale_line_points[0]
            p2 = self.scale_line_points[1]
            sx1 = int(p1[0] * scale) + ox
            sy1 = int(p1[1] * scale) + oy
            sx2 = int(p2[0] * scale) + ox
            sy2 = int(p2[1] * scale) + oy
            
            self.canvas.create_line(sx1, sy1, sx2, sy2, fill="red", width=2, dash=(5, 2))
            # Text label for it
            cx, cy = (sx1+sx2)/2, (sy1+sy2)/2
            if self.pixels_per_cm > 0:
                dist_cm = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2) / self.pixels_per_cm
                self.canvas.create_text(cx, cy, text=f"{dist_cm:.2f} cm", fill="red", anchor=tk.S, font=("Arial", 10, "bold"))

        # Temp drawing (Scale line or Active ROI)
        if hasattr(self, 'temp_points') and len(self.temp_points) > 0:
            t_points = []
            for p in self.temp_points:
                sx = int(p[0] * scale) + ox
                sy = int(p[1] * scale) + oy
                t_points.append((sx, sy))
                
            if self.mode == "scale" and len(t_points) == 1:
                # Dot for start
                self.canvas.create_oval(t_points[0][0]-3, t_points[0][1]-3, t_points[0][0]+3, t_points[0][1]+3, fill="red")
            
            elif self.mode == "roi":
                 # Draw lines between points
                 for i in range(len(t_points)):
                     px, py = t_points[i]
                     self.canvas.create_oval(px-3, py-3, px+3, py+3, fill="cyan", outline="blue")
                     if i > 0:
                         prev = t_points[i-1]
                         self.canvas.create_line(prev[0], prev[1], px, py, fill="cyan")

    def start_scale_mode(self):
        self.mode = "scale"
        self.temp_points = []
        # Removed popup, user knows what to do
        self.root.config(cursor="cross") # Ensure cursor indicates action
        self.show_image(self.cv_image_small) # Ensure we are viewing clean image

    def start_roi_mode(self):
        self.mode = "roi"
        self.temp_points = []
        self.roi_points = [] # Clear previous
        self.roi_mask = None
        self.show_image(self.cv_image_small) # Clear threshold preview if any
        # Removed popup
        self.root.config(cursor="cross")

    def reset_roi(self):
        self.roi_points = []
        self.roi_mask = None
        self.show_image(self.cv_image_small)

    def on_canvas_click(self, event):
        if self.cv_image is None: return
        
        # Convert screen coords to image coords
        cx = event.x
        cy = event.y
        
        # Convert screen coords to image coords method
        
        # Scale factor from Original -> Screen = (preview_scale * zoom_scale)
        scale = self.get_display_scale_factor()
        if scale == 0: return

        ox, oy = self.pan_offset_x, self.pan_offset_y
        
        img_x = int((cx - ox) / scale)
        img_y = int((cy - oy) / scale)
        
        # Bounds check
        if img_x < 0 or img_x >= self.original_w or img_y < 0 or img_y >= self.original_h:
            return

        if self.mode == "scale":
            self.temp_points.append((img_x, img_y))
            self.redraw_overlays()
            
            if len(self.temp_points) == 2:
                self.finish_scale()

        elif self.mode == "roi":
            self.temp_points.append((img_x, img_y))
            self.redraw_overlays()

    def on_canvas_rclick(self, event):
        if self.mode == "roi" and len(self.temp_points) > 2:
            self.roi_points = list(self.temp_points)
            self.temp_points = []
            self.mode = "view"
            self.create_roi_mask()
            self.redraw_overlays()
            self.calculate_area() # Update after ROI change

    def on_canvas_move(self, event):
        # Optional: rubber banding could go here
        pass

    def finish_scale(self):
        p1 = self.temp_points[0]
        p2 = self.temp_points[1]
        
        dist_px = np.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)
        
        if dist_px == 0:
            self.temp_points = []
            return

        # Ask user for real distance
        real_dist = simpledialog.askfloat("Scale Input", "Enter length in cm:", initialvalue=10.0)
        
        if real_dist and real_dist > 0:
            self.pixels_per_cm = dist_px / real_dist
            self.lbl_scale.config(text=f"Scale: {self.pixels_per_cm:.2f} px/cm", fg="green")
            self.mode = "view"
            self.scale_line_points = list(self.temp_points) # Store for persistent display
            self.temp_points = []
            self.redraw_overlays()
            # If area was already calculated, update it
            if self.var_otsu.get() or self.val_threshold.get() > 0:
                 self.calculate_area()
        else:
            self.temp_points = []
            self.redraw_overlays()

    def create_roi_mask(self):
        if not self.roi_points:
            self.roi_mask = None
            return
            
        mask = np.zeros((self.original_h, self.original_w), dtype=np.uint8)
        pts = np.array(self.roi_points, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(mask, [pts], 255)
        self.roi_mask = mask

    def toggle_otsu(self):
        if self.cv_image is None: return
        if self.var_otsu.get():
            self.calculate_exg_otsu()
        else:
            # Revert to manual slider value
            self.update_threshold_display(self.val_threshold.get())

    def on_threshold_result_entry(self):
        if self.cv_image is None: return
        self.var_otsu.set(False)
        self.update_threshold_display(self.val_threshold.get())
        
    def increment_threshold(self, delta):
        if self.cv_image is None: return
        current = self.val_threshold.get()
        new_val = round(current + delta, 1)
        # Clamp
        new_val = max(0.0, min(50.0, new_val))
        self.val_threshold.set(new_val)
        self.var_otsu.set(False)
        self.update_threshold_display(new_val)

    def on_threshold_change(self, value):
        if self.cv_image is None: return
        # Setting DoubleVar from scale string value
        val = float(value)
        self.val_threshold.set(val) 
        self.var_otsu.set(False) # Disable auto if manual moved
        self.update_threshold_display(val)

    def calculate_exg_otsu(self):
        # ExG = 2*G - R - B
        # Calc for small image first (for fast display)
        if self.cv_image_small is not None:
             img_float_s = self.cv_image_small.astype(np.float32)
             B_s, G_s, R_s = cv2.split(img_float_s)
             exg_small = 2 * G_s - R_s - B_s
             # Blur to allow floating point thresholding precision
             self.exg_map_small = cv2.GaussianBlur(exg_small, (5, 5), 0)
        
        # Calc for full image (for saving result)
        # We can do this lazily or now. Doing it now avoids lag when clicking "Save".
        # But if image is huge, maybe wait? 
        # Let's do it now, assuming 24MP is manageable once.
        img_float = self.cv_image.astype(np.float32)
        B, G, R = cv2.split(img_float)
        
        exg = 2 * G - R - B
        # Blur to allow floating point thresholding precision
        self.exg_map = cv2.GaussianBlur(exg, (5, 5), 0)
        
        # Simple method (on small map for speed):
        exg_uint8 = np.clip(self.exg_map_small, 0, 255).astype(np.uint8)
        
        # Otsu
        thresh_val_otsu, binary = cv2.threshold(exg_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        self.val_threshold.set(float(thresh_val_otsu))
        self.update_threshold_display(float(thresh_val_otsu))

    def update_threshold_display(self, thresh_val):
        if self.cv_image_small is None: return
        
        # 1. VISUALIZATION (Fast, using small image)
        if not hasattr(self, 'exg_map_small'):
             # fallback
             img_float_s = self.cv_image_small.astype(np.float32)
             B_s, G_s, R_s = cv2.split(img_float_s)
             self.exg_map_small = 2 * G_s - R_s - B_s

        # Create binary mask on small image
        binary_small = (self.exg_map_small > thresh_val).astype(np.uint8) * 255
        
        # Apply ROI mask if exists
        # ROI mask is full size. Need small version for display?
        # Or just generate small ROI mask on the fly.
        if self.roi_points:
             mask_s = np.zeros((self.cv_image_small.shape[0], self.cv_image_small.shape[1]), dtype=np.uint8)
             # Scale ROI points to small image
             pts_s = []
             for p in self.roi_points:
                 pts_s.append([int(p[0] * self.preview_scale), int(p[1] * self.preview_scale)])
             
             pts_s = np.array(pts_s, np.int32).reshape((-1, 1, 2))
             cv2.fillPoly(mask_s, [pts_s], 255)
             
             final_mask_small = cv2.bitwise_and(binary_small, binary_small, mask=mask_s)
        else:
             final_mask_small = binary_small
             mask_s = None

        # Create display composition
        display_vis = self.cv_image_small.copy()
        display_vis[final_mask_small == 255] = [0, 255, 0]
        
        # Darken outside ROI (Visualization only)
        if mask_s is not None:
             bg_mask = cv2.bitwise_not(mask_s)
             bg = cv2.bitwise_and(display_vis, display_vis, mask=bg_mask)
             bg = (bg * 0.3).astype(np.uint8)
             fg = cv2.bitwise_and(display_vis, display_vis, mask=mask_s)
             display_vis = cv2.add(fg, bg)
        
        self.show_image(display_vis)
        
        # 2. CALCULATION (Lazy or separate? Let's do it here but ONLY calculation)
        # We do NOT create full size visualization images anymore.
        
        # Use existing full size exg map for area calc
        # NOTE: If exg_map isn't ready (lazy loading?), create it.
        # For now assume it's created in calculate_exg_otsu or init.
        if hasattr(self, 'exg_map'):
             # This simple comparison is fast enough mostly.
             # If extremely large, even this might be slow, but much faster than cv2.addWeighted on full image.
             self.current_analysis_mask = (self.exg_map > thresh_val).astype(np.uint8) * 255
             
             if self.roi_mask is not None:
                 # Bitwise AND on boolean arrays or use OpenCV
                 # OpenCV bitwise_and on huge arrays can be heavy, but it's likely optimized.
                 self.current_analysis_mask = cv2.bitwise_and(self.current_analysis_mask, self.current_analysis_mask, mask=self.roi_mask)
             
        self.calculate_area_val() # Update number label

    def calculate_area(self):
        # Triggered by button, but we do it live usually. 
        # Just ensure we have valid masks
        if self.cv_image is None: return
        self.calculate_area_val()
        
    def calculate_area_val(self):
        if not hasattr(self, 'current_analysis_mask'): return
        
        # Count non-zero
        pixel_count = cv2.countNonZero(self.current_analysis_mask)
        
        if self.pixels_per_cm > 0:
            area_cm2 = pixel_count / (self.pixels_per_cm ** 2)
            self.lbl_area.config(text=f"Area: {area_cm2:.2f} cm²")
            self.last_area_cm2 = area_cm2
            self.unsaved_changes = True # Mark as needing save
        else:
            self.lbl_area.config(text=f"Area: {pixel_count} px (No Scale)")
            self.last_area_cm2 = None
    
    def save_results(self):
        if self.cv_image is None or not hasattr(self, 'last_area_cm2') or self.last_area_cm2 is None:
            messagebox.showwarning("Save Error", "No calculated area to save. Please load image, set scale, and adjust ExG.")
            return

        csv_file = "results.csv"
        file_exists = os.path.isfile(csv_file)
        
        try:
            with open(csv_file, mode='a', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                if not file_exists:
                    writer.writerow(["Timestamp", "File Path", "Leaf Area (cm^2)"])
                
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                writer.writerow([timestamp, self.current_file_path, f"{self.last_area_cm2:.4f}"])
            
            messagebox.showinfo("Saved", f"Results saved to {csv_file}")
            self.unsaved_changes = False
            
        except Exception as e:
            messagebox.showerror("Error", f"Could not save to CSV:\n{e}")

    def show_help(self):
        help_win = tk.Toplevel(self.root)
        help_win.title("How to Use - LAIZER")
        help_win.geometry("600x600")
        
        txt = scrolledtext.ScrolledText(help_win, wrap=tk.WORD, width=60, height=30)
        txt.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        readme_path = resource_path("README.md")
        
        try:
            with open(readme_path, "r", encoding="utf-8") as f:
                content = f.read()
            txt.insert(tk.END, content)
        except Exception as e:
            txt.insert(tk.END, f"Could not load README file.\n\nError: {e}")
            
        txt.config(state=tk.DISABLED) # Read-only

if __name__ == "__main__":
    try:
        # High DPI support for Windows
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except:
        pass

    root = tk.Tk()
    app = GreenLeafApp(root)
    root.mainloop()

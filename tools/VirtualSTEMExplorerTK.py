import os
import sys
import glob
import re
import json
from pathlib import Path
import numpy as np

import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.patches import Rectangle, Circle

import nbed


# --- Path & Config Management ---
def get_app_config_path() -> Path:
    if sys.platform == "win32":
        base_dir = Path(os.getenv("APPDATA", Path.home() / "AppData" / "Roaming"))
    else:
        base_dir = Path(os.getenv("XDG_CONFIG_HOME", Path.home() / ".config"))
    
    config_dir = base_dir / "VirtualSTEMExplorerTK"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir / "VirtualSTEMExplorerCTK_config.json"

CONFIG_FILE = get_app_config_path()

DEFAULT_CONFIG = {
    "theme": "dark",  # Options: "dark", "light"
    "cal_db_path": str(Path.home() / "MicroscopeCalibrationDB.json"),
    "left_cmap": "gray",
    "right_cmap": "turbo",
    "dx": 5,
    "dy": 5,
    "r_in": 0.0,
    "r_out": 25.0,
    "bin_scan_x": 2,
    "bin_scan_y": 2,
    "bin_det_x": 1,
    "bin_det_y": 1
}

def load_config():
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                cfg = json.load(f)
                for k, v in DEFAULT_CONFIG.items():
                    cfg.setdefault(k, v)
                return cfg
        except Exception as e:
            print(f"Warning: Could not read config from {CONFIG_FILE} ({e}). Using defaults.")
    return DEFAULT_CONFIG.copy()

def save_config(cfg):
    try:
        with open(CONFIG_FILE, 'w', encoding='utf-8') as f:
            json.dump(cfg, f, indent=4)
        print(f"Configuration saved to: {CONFIG_FILE}")
    except Exception as e:
        print(f"Failed to save configuration to {CONFIG_FILE}: {e}")


def style_matplotlib_toolbar(toolbar, theme_mode="dark"):
    """Styles the Matplotlib Tkinter toolbar widgets to match CustomTkinter appearance."""
    if theme_mode == "dark":
        bg_color = "#1a1a1a"
        fg_color = "#ffffff"
        btn_bg = "#2b2b2b"
        active_bg = "#3a3a3a"
    else:
        bg_color = "#ebebeb"
        fg_color = "#000000"
        btn_bg = "#dbdbdb"
        active_bg = "#c8c8c8"

    toolbar.configure(background=bg_color)
    for child in toolbar.winfo_children():
        if isinstance(child, tk.Button):
            try:
                child.configure(
                    background=btn_bg,
                    foreground=fg_color,
                    activebackground=active_bg,
                    activeforeground=fg_color,
                    relief=tk.FLAT,
                    borderwidth=0
                )
            except tk.TclError:
                pass
        elif isinstance(child, (tk.Label, tk.Frame)):
            try:
                child.configure(background=bg_color, foreground=fg_color)
            except tk.TclError:
                pass


# --- Modal Dialogs ---
class MetadataViewerDialog(ctk.CTkToplevel):
    """CustomTkinter Metadata Viewer dialog with live filter and clipboard copy."""
    def __init__(self, parent, metadata_dict, dataset_name="Dataset"):
        super().__init__(parent)
        self.title(f"Metadata Viewer - {dataset_name}")
        self.geometry("700x650")
        self.grab_set()

        self.metadata = metadata_dict if isinstance(metadata_dict, dict) else {}
        self.formatted_str = self._format_metadata(self.metadata)

        f_main = ctk.CTkFrame(self, corner_radius=10)
        f_main.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        # Header & Search Bar
        f_top = ctk.CTkFrame(f_main, fg_color="transparent")
        f_top.pack(fill=tk.X, padx=10, pady=(10, 6))

        ctk.CTkLabel(f_top, text="Metadata Entries:", font=("Arial", 13, "bold")).pack(side=tk.LEFT)

        self.var_filter = ctk.StringVar()
        self.var_filter.trace_add("write", self._apply_filter)
        
        entry_filter = ctk.CTkEntry(f_top, placeholder_text="Filter keys...", textvariable=self.var_filter, width=220)
        entry_filter.pack(side=tk.RIGHT)

        # Text Box Container
        self.txt_view = ctk.CTkTextbox(f_main, font=("Courier", 11), wrap="none")
        self.txt_view.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)

        # Bottom Bar
        f_btn = ctk.CTkFrame(f_main, fg_color="transparent")
        f_btn.pack(fill=tk.X, padx=10, pady=(4, 10))

        btn_copy = ctk.CTkButton(f_btn, text="📋 Copy to Clipboard", command=self._copy_to_clipboard, width=150)
        btn_copy.pack(side=tk.LEFT)

        btn_close = ctk.CTkButton(f_btn, text="Close", command=self.destroy, fg_color="transparent", border_width=1, width=90)
        btn_close.pack(side=tk.RIGHT)

        # Populate Text
        self._display_text(self.formatted_str)

    def _format_metadata(self, md) -> str:
        if not md:
            return "No metadata found or metadata dictionary is empty."
        try:
            return json.dumps(md, indent=2, default=str)
        except Exception:
            lines = []
            for k, v in md.items():
                lines.append(f"{k}: {v}")
            return "\n".join(lines)

    def _display_text(self, text_content):
        self.txt_view.configure(state="normal")
        self.txt_view.delete("1.0", tk.END)
        self.txt_view.insert("1.0", text_content)
        self.txt_view.configure(state="disabled")

    def _apply_filter(self, *args):
        query = self.var_filter.get().strip().lower()
        if not query:
            self._display_text(self.formatted_str)
            return

        filtered_lines = [
            line for line in self.formatted_str.splitlines() 
            if query in line.lower()
        ]
        self._display_text("\n".join(filtered_lines) if filtered_lines else f"No matches found for '{query}'.")

    def _copy_to_clipboard(self):
        text_to_copy = self.txt_view.get("1.0", tk.END).strip()
        self.clipboard_clear()
        self.clipboard_append(text_to_copy)
        messagebox.showinfo("Copied", "Metadata text copied to clipboard!", parent=self)


class ConfigDialog(ctk.CTkToplevel):
    """CustomTkinter preferences dialog."""
    def __init__(self, parent, current_cfg):
        super().__init__(parent)
        self.title("Preferences")
        self.resizable(False, False)
        self.grab_set()

        self.cfg = current_cfg.copy()
        self.updated = False

        f_main = ctk.CTkFrame(self, corner_radius=10)
        f_main.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        # 1. UI Appearance
        f_theme = ctk.CTkFrame(f_main)
        f_theme.pack(fill=tk.X, padx=10, pady=8)
        ctk.CTkLabel(f_theme, text="GUI Theme:", font=("Arial", 12, "bold")).pack(side=tk.LEFT, padx=10)
        self.var_theme = ctk.StringVar(value=self.cfg.get("theme", "dark"))
        cb_theme = ctk.CTkOptionMenu(f_theme, variable=self.var_theme, values=["dark", "light"], width=100)
        cb_theme.pack(side=tk.RIGHT, padx=10, pady=8)

        # 2. Calibration DB Path
        f_db = ctk.CTkFrame(f_main)
        f_db.pack(fill=tk.X, padx=10, pady=8)
        ctk.CTkLabel(f_db, text="Calibration DB Path:", font=("Arial", 12, "bold")).pack(anchor='w', padx=10, pady=(6,0))
        
        f_db_inner = ctk.CTkFrame(f_db, fg_color="transparent")
        f_db_inner.pack(fill=tk.X, padx=10, pady=6)
        self.var_db_path = ctk.StringVar(value=self.cfg.get("cal_db_path", ""))
        ctk.CTkEntry(f_db_inner, textvariable=self.var_db_path, width=320).pack(side=tk.LEFT, padx=(0, 6))
        ctk.CTkButton(f_db_inner, text="Browse...", width=80, command=self.browse_db).pack(side=tk.LEFT)

        # 3. Display Defaults
        f_disp = ctk.CTkFrame(f_main)
        f_disp.pack(fill=tk.X, padx=10, pady=8)
        cmap_opts = ['gray', 'gray_r', 'inferno', 'magma', 'viridis', 'plasma', 'turbo', 'cividis', 'hsv', 'twilight']

        row1 = ctk.CTkFrame(f_disp, fg_color="transparent")
        row1.pack(fill=tk.X, padx=10, pady=4)
        ctk.CTkLabel(row1, text="Left Cmap:").pack(side=tk.LEFT, padx=4)
        self.var_l_cmap = ctk.StringVar(value=self.cfg.get("left_cmap", "gray"))
        ctk.CTkOptionMenu(row1, variable=self.var_l_cmap, values=cmap_opts, width=110).pack(side=tk.LEFT, padx=4)

        ctk.CTkLabel(row1, text="Right Cmap:").pack(side=tk.LEFT, padx=(12, 4))
        self.var_r_cmap = ctk.StringVar(value=self.cfg.get("right_cmap", "turbo"))
        ctk.CTkOptionMenu(row1, variable=self.var_r_cmap, values=cmap_opts, width=110).pack(side=tk.LEFT, padx=4)

        row2 = ctk.CTkFrame(f_disp, fg_color="transparent")
        row2.pack(fill=tk.X, padx=10, pady=4)
        ctk.CTkLabel(row2, text="Default dx/dy:").pack(side=tk.LEFT, padx=4)
        self.var_dx = ctk.StringVar(value=str(self.cfg.get("dx", 5)))
        self.var_dy = ctk.StringVar(value=str(self.cfg.get("dy", 5)))
        ctk.CTkEntry(row2, textvariable=self.var_dx, width=45).pack(side=tk.LEFT, padx=2)
        ctk.CTkEntry(row2, textvariable=self.var_dy, width=45).pack(side=tk.LEFT, padx=2)

        ctk.CTkLabel(row2, text="Default r_in/out:").pack(side=tk.LEFT, padx=(12, 4))
        self.var_rin = ctk.StringVar(value=str(self.cfg.get("r_in", 0.0)))
        self.var_rout = ctk.StringVar(value=str(self.cfg.get("r_out", 25.0)))
        ctk.CTkEntry(row2, textvariable=self.var_rin, width=45).pack(side=tk.LEFT, padx=2)
        ctk.CTkEntry(row2, textvariable=self.var_rout, width=45).pack(side=tk.LEFT, padx=2)

        # Action Buttons
        f_btn = ctk.CTkFrame(f_main, fg_color="transparent")
        f_btn.pack(pady=12)
        ctk.CTkButton(f_btn, text="Save & Apply", command=self.on_save, width=110).pack(side=tk.LEFT, padx=6)
        ctk.CTkButton(f_btn, text="Cancel", command=self.destroy, fg_color="transparent", border_width=1, width=90).pack(side=tk.LEFT, padx=6)

    def browse_db(self):
        fn = filedialog.askopenfilename(
            title="Select Microscope Calibration JSON File",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        if fn:
            self.var_db_path.set(fn)

    def on_save(self):
        try:
            self.cfg["theme"] = self.var_theme.get()
            self.cfg["cal_db_path"] = self.var_db_path.get().strip()
            self.cfg["left_cmap"] = self.var_l_cmap.get()
            self.cfg["right_cmap"] = self.var_r_cmap.get()
            self.cfg["dx"] = int(self.var_dx.get())
            self.cfg["dy"] = int(self.var_dy.get())
            self.cfg["r_in"] = float(self.var_rin.get())
            self.cfg["r_out"] = float(self.var_rout.get())

            save_config(self.cfg)
            self.updated = True
            self.destroy()
        except ValueError as e:
            messagebox.showerror("Validation Error", f"Please check numerical input fields:\n{e}", parent=self)


class FileImportDialog(ctk.CTkToplevel):
    """CustomTkinter File Import dialog."""
    def __init__(self, parent, initial_cfg):
        super().__init__(parent)
        self.title("4D-STEM Dataset Import Options")
        self.resizable(False, False)
        self.grab_set()

        self.result = None

        f_main = ctk.CTkFrame(self, corner_radius=10)
        f_main.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        ctk.CTkLabel(f_main, text="Select 4D-STEM Dataset File:", font=("Arial", 12, "bold")).pack(anchor='w', padx=10, pady=(6, 2))
        
        f_path = ctk.CTkFrame(f_main, fg_color="transparent")
        f_path.pack(fill=tk.X, padx=10, pady=4)
        self.var_filepath = ctk.StringVar()
        ctk.CTkEntry(f_path, textvariable=self.var_filepath, width=380).pack(side=tk.LEFT, padx=(0, 6))
        ctk.CTkButton(f_path, text="Browse...", width=80, command=self.browse_file).pack(side=tk.LEFT)

        f_fmt = ctk.CTkFrame(f_main, fg_color="transparent")
        f_fmt.pack(fill=tk.X, padx=10, pady=6)
        ctk.CTkLabel(f_fmt, text="Data Format:").pack(side=tk.LEFT, padx=(0, 6))
        self.var_format = ctk.StringVar(value="Auto-detect")
        ctk.CTkOptionMenu(f_fmt, variable=self.var_format, values=["Auto-detect", "DECTRIS", "PantaRhei"], width=130).pack(side=tk.LEFT)

        f_bin = ctk.CTkFrame(f_main)
        f_bin.pack(fill=tk.X, padx=10, pady=10)
        ctk.CTkLabel(f_bin, text="Binning Settings", font=("Arial", 11, "bold")).pack(anchor='w', padx=10, pady=(6,2))

        b_row1 = ctk.CTkFrame(f_bin, fg_color="transparent")
        b_row1.pack(fill=tk.X, padx=10, pady=4)
        ctk.CTkLabel(b_row1, text="Real Space (bin_scan Y, X):").pack(side=tk.LEFT, padx=(0, 6))
        self.var_bs_y = ctk.StringVar(value=str(initial_cfg.get("bin_scan_y", 2)))
        self.var_bs_x = ctk.StringVar(value=str(initial_cfg.get("bin_scan_x", 2)))
        ctk.CTkEntry(b_row1, textvariable=self.var_bs_y, width=45).pack(side=tk.LEFT, padx=2)
        ctk.CTkEntry(b_row1, textvariable=self.var_bs_x, width=45).pack(side=tk.LEFT, padx=2)

        b_row2 = ctk.CTkFrame(f_bin, fg_color="transparent")
        b_row2.pack(fill=tk.X, padx=10, pady=4)
        ctk.CTkLabel(b_row2, text="Reciprocal Space (bin_det Y, X):").pack(side=tk.LEFT, padx=(0, 6))
        self.var_bd_y = ctk.StringVar(value=str(initial_cfg.get("bin_det_y", 1)))
        self.var_bd_x = ctk.StringVar(value=str(initial_cfg.get("bin_det_x", 1)))
        ctk.CTkEntry(b_row2, textvariable=self.var_bd_y, width=45).pack(side=tk.LEFT, padx=2)
        ctk.CTkEntry(b_row2, textvariable=self.var_bd_x, width=45).pack(side=tk.LEFT, padx=2)

        f_btn = ctk.CTkFrame(f_main, fg_color="transparent")
        f_btn.pack(pady=10)
        ctk.CTkButton(f_btn, text="Load Data", command=self.on_ok, width=110).pack(side=tk.LEFT, padx=6)
        ctk.CTkButton(f_btn, text="Cancel", command=self.destroy, fg_color="transparent", border_width=1, width=90).pack(side=tk.LEFT, padx=6)

    def auto_detect_format(self, filepath: str) -> str:
        fn_lower = filepath.lower()
        if fn_lower.endswith(('.prz', '.prz.zip')) or 'pantarhei' in fn_lower:
            return 'PantaRhei'
        elif fn_lower.endswith(('.h5', '.hdf5')):
            return 'DECTRIS'
        elif fn_lower.endswith('.zip') or fn_lower.endswith('.npz'):
            return 'PantaRhei'
        return 'DECTRIS'

    def browse_file(self):
        filename = filedialog.askopenfilename(
            title="Select 4D-STEM Dataset File",
            filetypes=[
                ("All Supported Formats", "*.h5 *.hdf5 *.prz *.zip *.npy *.npz"),
                ("DECTRIS HDF5 Files", "*.h5 *.hdf5"),
                ("PantaRhei NumPy Archives", "*.prz *.zip *.npy *.npz"),
                ("All Files", "*.*")
            ]
        )
        if filename:
            self.var_filepath.set(filename)
            self.var_format.set(self.auto_detect_format(filename))

    def on_ok(self):
        filepath = self.var_filepath.get().strip()
        if not filepath or not os.path.exists(filepath):
            messagebox.showerror("Invalid File", "Please select a valid dataset file.", parent=self)
            return

        try:
            bs_y = int(self.var_bs_y.get())
            bs_x = int(self.var_bs_x.get())
            bd_y = int(self.var_bd_y.get())
            bd_x = int(self.var_bd_x.get())
        except ValueError:
            messagebox.showerror("Invalid Binning", "Binning factors must be integer values.", parent=self)
            return

        chosen_fmt = self.var_format.get()
        if chosen_fmt == "Auto-detect":
            chosen_fmt = self.auto_detect_format(filepath)

        fn_lower = filepath.lower()
        is_zip = fn_lower.endswith('.zip') or fn_lower.endswith('.prz.zip')

        filedir, filename = os.path.split(filepath)
        basename, suffix = os.path.splitext(filename)

        self.result = {
            "full_path": filepath,
            "path": filedir + "/",
            "filebasename": basename,
            "filesuffix": suffix,
            "format_type": chosen_fmt,
            "is_zipfile": is_zip,
            "bin_scan": (bs_y, bs_x),
            "bin_det": (bd_y, bd_x)
        }
        self.destroy()


class ManualSamplingDialog(ctk.CTkToplevel):
    """CustomTkinter Manual Sampling Override dialog."""
    def __init__(self, parent, current_re_samp, current_rec_samp):
        super().__init__(parent)
        self.title("Manual Sampling Calibration")
        self.resizable(False, False)
        self.grab_set()

        self.result = None

        f_main = ctk.CTkFrame(self, corner_radius=10)
        f_main.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        ctk.CTkLabel(f_main, text="Override pixel sampling calibrations:", font=("Arial", 12, "bold")).pack(anchor='w', padx=10, pady=(6, 10))

        row1 = ctk.CTkFrame(f_main, fg_color="transparent")
        row1.pack(fill=tk.X, padx=10, pady=4)
        ctk.CTkLabel(row1, text="Real-Space (re_samp):").pack(side=tk.LEFT, padx=(0, 6))
        self.var_re_samp = ctk.StringVar(value=f"{current_re_samp:.6g}")
        ctk.CTkEntry(row1, textvariable=self.var_re_samp, width=100).pack(side=tk.LEFT, padx=4)
        ctk.CTkLabel(row1, text="nm / px").pack(side=tk.LEFT)

        row2 = ctk.CTkFrame(f_main, fg_color="transparent")
        row2.pack(fill=tk.X, padx=10, pady=4)
        ctk.CTkLabel(row2, text="Reciprocal-Space (rec_samp):").pack(side=tk.LEFT, padx=(0, 6))
        self.var_rec_samp = ctk.StringVar(value=f"{current_rec_samp:.6g}")
        ctk.CTkEntry(row2, textvariable=self.var_rec_samp, width=100).pack(side=tk.LEFT, padx=4)
        ctk.CTkLabel(row2, text="1/nm / px").pack(side=tk.LEFT)

        f_btn = ctk.CTkFrame(f_main, fg_color="transparent")
        f_btn.pack(pady=12)
        ctk.CTkButton(f_btn, text="Apply & Update Plot", command=self.on_apply, width=140).pack(side=tk.LEFT, padx=6)
        ctk.CTkButton(f_btn, text="Cancel", command=self.destroy, fg_color="transparent", border_width=1, width=80).pack(side=tk.LEFT, padx=6)

    def on_apply(self):
        try:
            val_re = float(self.var_re_samp.get())
            val_rec = float(self.var_rec_samp.get())
            if val_re <= 0 or val_rec <= 0:
                raise ValueError("Sampling values must be strictly positive.")

            self.result = {"re_samp": val_re, "rec_samp": val_rec}
            self.destroy()
        except ValueError as e:
            messagebox.showerror("Invalid Input", f"Please enter valid positive numbers:\n{e}", parent=self)


# --- Main Application Explorer ---
def VirtualSTEMExplorerCTK(myset=None, initial_filepath=None, initial_format=None):
    config = load_config()
    target_dict = {}

    theme_mode = config.get("theme", "dark")
    ctk.set_appearance_mode(theme_mode)
    ctk.set_default_color_theme("blue")

    root = ctk.CTk()
    root.title("VirtualSTEMExplorerTK [CustomTkinter]")
    root.geometry("1420x960")

    if theme_mode == "dark":
        plt.style.use('dark_background')
    else:
        plt.style.use('default')

    def show_about_dialog():
        messagebox.showinfo(
            "About VirtualSTEMExplorerTK",
            "VirtualSTEMExplorerTK v2.3 (CustomTkinter)\n"
            "Interactive 4D-STEM Analysis Suite\n\n"
            "Copyright © 2026 L. Houben - Weizmann Institute of Science."
        )

    state = {
        'myset': myset,
        'path': "",
        'filebasename': "VirtualSTEMExplorer_Export",
        'filesuffix': ".h5",
        'format_type': "DECTRIS",
        're_samp': 1.0,
        'rec_samp': 1.0,
        'x': 0, 'y': 0,
        'dx': config["dx"], 'dy': config["dy"],
        'cqx': 0, 'cqy': 0,
        'r_in': config["r_in"], 'r_out': config["r_out"],
        'calc_mode': 'Sum',
        'left_cmap': config["left_cmap"],
        'right_cmap': config["right_cmap"]
    }

    ruler1_state = {'active': False, 'p1': None, 'p2': None, 'length': 0.0}
    ruler2_state = {'active': False, 'p1': None, 'p2': None, 'dq': 0.0, 'd_spacing': 0.0}

    # Top Control Bar
    top_bar = ctk.CTkFrame(root, height=45, corner_radius=0)
    top_bar.pack(side=tk.TOP, fill=tk.X)

    btn_open = ctk.CTkButton(top_bar, text="📁 Open Dataset...", width=130, command=lambda: open_file_dialog_action())
    btn_open.pack(side=tk.LEFT, padx=8, pady=8)

    btn_config = ctk.CTkButton(top_bar, text="⚙️ Settings...", width=110, fg_color="transparent", border_width=1, command=lambda: open_config_dialog_action())
    btn_config.pack(side=tk.LEFT, padx=4, pady=8)

    lbl_status = ctk.CTkLabel(top_bar, text="No dataset loaded. Click 'Open Dataset' to start.", font=("Arial", 11, "italic"))
    lbl_status.pack(side=tk.LEFT, padx=12, pady=8)

    canvas_frame = ctk.CTkFrame(root)
    canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=(4, 0))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.subplots_adjust(top=0.88, bottom=0.15, left=0.08, right=0.92, wspace=0.35)

    canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    toolbar = NavigationToolbar2Tk(canvas, canvas_frame)
    style_matplotlib_toolbar(toolbar, theme_mode)
    toolbar.update()

    artists = {}

    def get_reciprocal_extent(center_x, center_y):
        x_min = -center_x * state['rec_samp']
        x_max = (state['detx'] - center_x) * state['rec_samp']
        y_min = -center_y * state['rec_samp']
        y_max = (state['dety'] - center_y) * state['rec_samp']
        return [x_min, x_max, y_min, y_max]

    def get_next_filename():
        save_prefix = os.path.join(state['path'], state['filebasename'] + "_export")
        save_fmt = "pdf"
        pattern = f"{save_prefix}_[0-9][0-9][0-9].{save_fmt}"
        existing_files = glob.glob(pattern)
        max_idx = 0
        for fname in existing_files:
            match = re.search(r'_(\d{3})\.' + re.escape(save_fmt) + r'$', fname)
            if match:
                idx = int(match.group(1))
                if idx > max_idx:
                    max_idx = idx
        next_idx = max_idx + 1
        return next_idx, f"{save_prefix}_{next_idx:03d}.{save_fmt}"

    def sync_target_dict():
        target_dict['x'] = int(state['x'])
        target_dict['y'] = int(state['y'])
        target_dict['x_phys'] = float(state['x'] * state['re_samp'])
        target_dict['y_phys'] = float(state['y'] * state['re_samp'])
        target_dict['re_samp'] = float(state['re_samp'])
        target_dict['rec_samp'] = float(state['rec_samp'])
        target_dict['dx'] = int(state['dx'])
        target_dict['dy'] = int(state['dy'])
        target_dict['cqx'] = float(state['cqx'])
        target_dict['cqy'] = float(state['cqy'])
        target_dict['r_in'] = float(state['r_in'])
        target_dict['r_out'] = float(state['r_out'])
        target_dict['r_in_phys'] = float(state['r_in'] * state['rec_samp'])
        target_dict['r_out_phys'] = float(state['r_out'] * state['rec_samp'])
        target_dict['left_cmap'] = state['left_cmap']
        target_dict['right_cmap'] = state['right_cmap']
        target_dict['mode'] = state['calc_mode']
        target_dict['index'] = int(state['y'] * state['scanx'] + state['x'])
        target_dict['ruler_real_dr'] = float(ruler1_state['length'])
        target_dict['ruler_recip_dq'] = float(ruler2_state['dq'])
        target_dict['ruler_recip_d'] = float(ruler2_state['d_spacing'])

    def compute_virtual_image(center_qx, center_qy, radius_in, radius_out, mode='Sum'):
        dety, detx = state['myset'].dim[2], state['myset'].dim[3]
        qy_2d, qx_2d = np.mgrid[:dety, :detx]
        
        dist_sq = (qx_2d - center_qx)**2 + (qy_2d - center_qy)**2
        mask = (dist_sq >= radius_in**2) & (dist_sq <= radius_out**2)
        
        if np.any(mask):
            masked_data = state['myset'].data[:, :, mask]
            if mode == 'Sum':
                vimage = np.sum(masked_data, axis=-1)
            elif mode == 'Variance':
                vimage = np.var(masked_data, axis=-1)
            elif mode == 'Fluctuation':
                mean_val = np.mean(masked_data, axis=-1)
                var_val = np.var(masked_data, axis=-1)
                vimage = np.where(mean_val > 1e-6, var_val / (mean_val**2 + 1e-6), 0.0)
            elif mode in ['COM Mag', 'COM Azimuth']:
                rel_qx = (qx_2d - center_qx)[mask]
                rel_qy = (qy_2d - center_qy)[mask]
                I_total = np.sum(masked_data, axis=-1)
                I_total_safe = np.where(I_total > 1e-6, I_total, 1e-6)
                com_x = np.sum(masked_data * rel_qx, axis=-1) / I_total_safe
                com_y = np.sum(masked_data * rel_qy, axis=-1) / I_total_safe
                if mode == 'COM Mag':
                    vimage = np.hypot(com_x, com_y) * state['rec_samp']
                else:
                    vimage = np.arctan2(com_y, com_x)
            else:
                vimage = np.sum(masked_data, axis=-1)
        else:
            vimage = np.zeros((state['scany'], state['scanx']))
        return vimage, mask

    def extract_avg_diffraction(center_y, center_x, radius_y, radius_x):
        y_min = max(0, center_y - radius_y)
        y_max = min(state['scany'], center_y + radius_y + 1)
        x_min = max(0, center_x - radius_x)
        x_max = min(state['scanx'], center_x + radius_x + 1)
        avg_slice = np.mean(state['myset'].data[y_min:y_max, x_min:x_max, :, :], axis=(0, 1))
        return np.log(avg_slice + 1.0)

    def apply_contrast(data, clip_percentiles, gamma):
        p_low, p_high = clip_percentiles
        vmin, vmax = np.percentile(data, [p_low, p_high])
        if vmax == vmin:
            vmax = vmin + 1e-5
        norm = np.clip((data - vmin) / (vmax - vmin), 0.0, 1.0)
        return np.power(norm, gamma)

    def update_left():
        if state['myset'] is None: return
        c_low = float(var_l_clip_low.get())
        c_high = float(var_l_clip_high.get())
        gamma = float(var_l_gamma.get())
        processed = apply_contrast(state['data_left_raw'], (c_low, c_high), gamma)
        artists['im1'].set_data(processed)
        canvas.draw_idle()

    def update_right():
        if state['myset'] is None: return
        c_low = float(var_r_clip_low.get())
        c_high = float(var_r_clip_high.get())
        gamma = float(var_r_gamma.get())
        processed = apply_contrast(state['data_right_raw'], (c_low, c_high), gamma)
        artists['im2'].set_data(processed)
        canvas.draw_idle()

    def refresh_detector_mask():
        if state['myset'] is None: return
        try:
            state['r_in'] = max(0.0, float(var_rin.get()))
            state['r_out'] = max(state['r_in'], float(var_rout.get()))
        except ValueError:
            return

        sync_target_dict()
        artists['im2'].set_extent(get_reciprocal_extent(state['cqx'], state['cqy']))
        artists['circle_in'].set_radius(state['r_in'] * state['rec_samp'])
        artists['circle_out'].set_radius(state['r_out'] * state['rec_samp'])

        state['data_left_raw'], _ = compute_virtual_image(state['cqx'], state['cqy'], state['r_in'], state['r_out'], mode=state['calc_mode'])
        ax1.set_title(f"Virtual Image [{state['calc_mode']}]", pad=15)
        update_left()

    def refresh_spatial_roi():
        if state['myset'] is None: return
        try:
            state['dx'] = max(0, int(var_dx.get()))
            state['dy'] = max(0, int(var_dy.get()))
        except ValueError:
            return

        sync_target_dict()
        artists['roi_rect'].set_bounds((state['x'] - state['dx'] - 0.5) * state['re_samp'],
                                       (state['y'] - state['dy'] - 0.5) * state['re_samp'],
                                       (2*state['dx'] + 1) * state['re_samp'],
                                       (2*state['dy'] + 1) * state['re_samp'])
        state['data_right_raw'] = extract_avg_diffraction(state['y'], state['x'], state['dy'], state['dx'])
        ax2.set_title(f"Diffraction Frame ({state['y']}, {state['x']})", pad=15)
        update_right()

    def rebuild_plot():
        if state['myset'] is None: return
        ax1.clear(); ax2.clear()

        scany, scanx = state['myset'].dim[0], state['myset'].dim[1]
        dety, detx = state['myset'].dim[2], state['myset'].dim[3]
        state['scany'] = scany; state['scanx'] = scanx
        state['dety'] = dety; state['detx'] = detx

        if state['x'] == 0 and state['y'] == 0:
            state['x'] = scanx // 2
            state['y'] = scany // 2
        if state['cqx'] == 0 and state['cqy'] == 0:
            state['cqx'] = detx // 2
            state['cqy'] = dety // 2

        extent_left = [0, scanx * state['re_samp'], 0, scany * state['re_samp']]
        extent_right = get_reciprocal_extent(state['cqx'], state['cqy'])

        state['data_left_raw'], _ = compute_virtual_image(state['cqx'], state['cqy'], state['r_in'], state['r_out'], mode=state['calc_mode'])
        state['data_right_raw'] = extract_avg_diffraction(state['y'], state['x'], state['dy'], state['dx'])

        img1_proc = apply_contrast(state['data_left_raw'], (0, 100), 1.0)
        img2_proc = apply_contrast(state['data_right_raw'], (0, 100), 1.0)

        # LEFT Panel
        artists['im1'] = ax1.imshow(img1_proc, cmap=state['left_cmap'], origin="lower", vmin=0, vmax=1, extent=extent_left)
        ax1.set_title(f"Virtual Image [{state['calc_mode']}]", pad=15)
        ax1.set_xlabel("x (nm)"); ax1.set_ylabel("y (nm)")

        sec_ax1_x = ax1.secondary_xaxis('top', functions=(lambda val: val / state['re_samp'], lambda val: val * state['re_samp']))
        sec_ax1_y = ax1.secondary_yaxis('right', functions=(lambda val: val / state['re_samp'], lambda val: val * state['re_samp']))
        sec_ax1_x.set_xlabel("x (pixels)", labelpad=4)
        sec_ax1_y.set_ylabel("y (pixels)", labelpad=4)

        px_x, px_y = state['x'] * state['re_samp'], state['y'] * state['re_samp']
        artists['v_line'] = ax1.axvline(px_x, color='yellow', lw=1)
        artists['h_line'] = ax1.axhline(px_y, color='yellow', lw=1)
        artists['roi_rect'] = Rectangle(((state['x'] - state['dx'] - 0.5) * state['re_samp'],
                                         (state['y'] - state['dy'] - 0.5) * state['re_samp']),
                                        (2*state['dx'] + 1) * state['re_samp'],
                                        (2*state['dy'] + 1) * state['re_samp'],
                                        linewidth=1, edgecolor='yellow', facecolor='none', linestyle='--')
        ax1.add_patch(artists['roi_rect'])

        # RIGHT Panel
        artists['im2'] = ax2.imshow(img2_proc, cmap=state['right_cmap'], origin="lower", vmin=0, vmax=1, extent=extent_right)
        ax2.set_title(f"Diffraction Frame ({state['y']}, {state['x']})", pad=15)
        ax2.set_xlabel("q_x (1/nm)"); ax2.set_ylabel("q_y (1/nm)")

        sec_ax2_x = ax2.secondary_xaxis('top', functions=(lambda val: val / state['rec_samp'] + state['cqx'], lambda val: (val - state['cqx']) * state['rec_samp']))
        sec_ax2_y = ax2.secondary_yaxis('right', functions=(lambda val: val / state['rec_samp'] + state['cqy'], lambda val: (val - state['cqy']) * state['rec_samp']))
        sec_ax2_x.set_xlabel("q_x (pixels)", labelpad=4)
        sec_ax2_y.set_ylabel("q_y (pixels)", labelpad=4)

        artists['center_marker'] = ax2.plot(0, 0, 'rx', markersize=8)[0]
        artists['circle_in'] = Circle((0, 0), state['r_in'] * state['rec_samp'], color='yellow', fill=False, linestyle='--', linewidth=1.5)
        artists['circle_out'] = Circle((0, 0), state['r_out'] * state['rec_samp'], color='yellow', fill=False, linestyle='-', linewidth=1.5)
        ax2.add_patch(artists['circle_in'])
        ax2.add_patch(artists['circle_out'])

        artists['ruler1_line'] = ax1.plot([], [], 'm--', lw=1.5, marker='o', markersize=4)[0]
        artists['ruler1_text'] = ax1.text(0.03, 0.95, "", transform=ax1.transAxes, color='magenta',
                                           fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="black", ec="magenta", alpha=0.8))
        artists['ruler1_text'].set_visible(False)

        artists['ruler2_line'] = ax2.plot([], [], 'c--', lw=1.5, marker='o', markersize=4)[0]
        artists['ruler2_text'] = ax2.text(0.03, 0.95, "", transform=ax2.transAxes, color='cyan',
                                           fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="black", ec="cyan", alpha=0.8))
        artists['ruler2_text'].set_visible(False)

        next_idx, _ = get_next_filename()
        btn_save.configure(text=f"Export Fig #{next_idx:03d}")
        sync_target_dict()
        canvas.draw()

    def align_diffraction_centers():
        """Finds the strongest peak in each frame, fits a 2D surface to centers, and applies integer shifts."""
        if state['myset'] is None:
            messagebox.showwarning("No Dataset Loaded", "Please load a dataset before running center alignment.")
            return

        print("\n--- Starting Diffraction Pattern Alignment ---")
        data_4d = state['myset'].data  # Shape: (scany, scanx, dety, detx)
        sy, sx, dy, dx = data_4d.shape

        print("1. Computing peak positions across all diffraction frames...")
        flat_data = data_4d.reshape(sy, sx, -1)
        max_flat_idx = np.argmax(flat_data, axis=-1)
        cy_map, cx_map = np.unravel_index(max_flat_idx, (dy, dx))

        # 2D Linear Regression surface: z = a0 + a1*x + a2*y
        y_coords, x_coords = np.mgrid[:sy, :sx]
        A = np.column_stack([np.ones(sy * sx), x_coords.ravel(), y_coords.ravel()])

        coeff_y, _, _, _ = np.linalg.lstsq(A, cy_map.ravel(), rcond=None)
        fitted_cy = (A @ coeff_y).reshape(sy, sx)

        coeff_x, _, _, _ = np.linalg.lstsq(A, cx_map.ravel(), rcond=None)
        fitted_cx = (A @ coeff_x).reshape(sy, sx)

        target_cy = int(round(np.mean(fitted_cy)))
        target_cx = int(round(np.mean(fitted_cx)))

        shift_y = np.round(target_cy - fitted_cy).astype(int)
        shift_x = np.round(target_cx - fitted_cx).astype(int)

        res_y = cy_map - fitted_cy
        res_x = cx_map - fitted_cx
        max_shift = int(np.max(np.hypot(shift_x, shift_y)))
        max_res = float(np.max(np.hypot(res_x, res_y)))

        print("--- Shift & Fit Statistics ---")
        print(f"  Target Center (cy, cx)          : ({target_cy}, {target_cx}) px")
        print(f"  Maximum Linear Shift Magnitude  : {max_shift} px")
        print(f"  Maximum Fit Residual Magnitude  : {max_res:.3f} px")
        print(f"  Mean Fit Residual (cy, cx)      : ({np.mean(np.abs(res_y)):.3f}, {np.mean(np.abs(res_x)):.3f}) px")

        print("2. Applying integer shifts to diffraction frames...")
        for y_idx in range(sy):
            for x_idx in range(sx):
                dy_s = shift_y[y_idx, x_idx]
                dx_s = shift_x[y_idx, x_idx]
                if dy_s != 0 or dx_s != 0:
                    data_4d[y_idx, x_idx, :, :] = np.roll(
                        data_4d[y_idx, x_idx, :, :], 
                        shift=(dy_s, dx_s), 
                        axis=(0, 1)
                    )

        state['cqy'] = target_cy
        state['cqx'] = target_cx
        
        rebuild_plot()
        print("--- Alignment Complete & Plots Updated ---\n")
        messagebox.showinfo(
            "Alignment Complete", 
            f"Diffraction frames aligned successfully!\n\n"
            f"Target Center: ({target_cy}, {target_cx}) px\n"
            f"Max Linear Shift: {max_shift} px\n"
            f"Max Residuum Shift: {max_res:.2f} px"
        )

    # Mouse Handlers
    def onclick(event):
        if state['myset'] is None: return
        if event.inaxes == ax1:
            if ruler1_state['active']:
                pt = (event.xdata, event.ydata)
                if ruler1_state['p1'] is None or ruler1_state['p2'] is not None:
                    ruler1_state['p1'] = pt
                    ruler1_state['p2'] = None
                    artists['ruler1_line'].set_data([pt[0]], [pt[1]])
                    artists['ruler1_text'].set_text(f"P1: ({pt[0]:.2f}, {pt[1]:.2f})\nClick P2...")
                else:
                    ruler1_state['p2'] = pt
                    p1 = ruler1_state['p1']
                    artists['ruler1_line'].set_data([p1[0], pt[0]], [p1[1], pt[1]])
                    dr = np.hypot(pt[0] - p1[0], pt[1] - p1[1])
                    ruler1_state['length'] = dr
                    sync_target_dict()
                    artists['ruler1_text'].set_text(f"Δr = {dr:.3f} nm\n({dr/state['re_samp']:.1f} px)")
                canvas.draw_idle()
            else:
                click_x = int(np.clip(event.xdata / state['re_samp'], 0, state['scanx'] - 1))
                click_y = int(np.clip(event.ydata / state['re_samp'], 0, state['scany'] - 1))
                state['x'], state['y'] = click_x, click_y
                artists['v_line'].set_xdata([state['x'] * state['re_samp'], state['x'] * state['re_samp']])
                artists['h_line'].set_ydata([state['y'] * state['re_samp'], state['y'] * state['re_samp']])
                refresh_spatial_roi()

        elif event.inaxes == ax2:
            if ruler2_state['active']:
                pt = (event.xdata, event.ydata)
                if ruler2_state['p1'] is None or ruler2_state['p2'] is not None:
                    ruler2_state['p1'] = pt
                    ruler2_state['p2'] = None
                    artists['ruler2_line'].set_data([pt[0]], [pt[1]])
                    artists['ruler2_text'].set_text(f"P1: ({pt[0]:.2f}, {pt[1]:.2f})\nClick P2...")
                else:
                    ruler2_state['p2'] = pt
                    p1 = ruler2_state['p1']
                    artists['ruler2_line'].set_data([p1[0], pt[0]], [p1[1], pt[1]])
                    dq = np.hypot(pt[0] - p1[0], pt[1] - p1[1])
                    d_spacing = 1.0 / dq if dq > 1e-6 else np.inf
                    ruler2_state['dq'] = dq
                    ruler2_state['d_spacing'] = d_spacing
                    sync_target_dict()
                    artists['ruler2_text'].set_text(f"Δq = {dq:.4f} 1/nm\nd = {d_spacing:.4f} nm")
                canvas.draw_idle()
            else:
                click_qx = event.xdata / state['rec_samp'] + state['cqx']
                click_qy = event.ydata / state['rec_samp'] + state['cqy']
                state['cqx'] = float(np.clip(click_qx, 0, state['detx'] - 1))
                state['cqy'] = float(np.clip(click_qy, 0, state['dety'] - 1))
                refresh_detector_mask()

    def onmove(event):
        if state['myset'] is None: return
        if ruler1_state['active'] and ruler1_state['p1'] is not None and ruler1_state['p2'] is None:
            if event.inaxes == ax1:
                p1 = ruler1_state['p1']
                p2 = (event.xdata, event.ydata)
                artists['ruler1_line'].set_data([p1[0], p2[0]], [p1[1], p2[1]])
                dr = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
                ruler1_state['length'] = dr
                sync_target_dict()
                artists['ruler1_text'].set_text(f"Δr = {dr:.3f} nm\n({dr/state['re_samp']:.1f} px)")
                canvas.draw_idle()

        if ruler2_state['active'] and ruler2_state['p1'] is not None and ruler2_state['p2'] is None:
            if event.inaxes == ax2:
                p1 = ruler2_state['p1']
                p2 = (event.xdata, event.ydata)
                artists['ruler2_line'].set_data([p1[0], p2[0]], [p1[1], p2[1]])
                dq = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
                d_spacing = 1.0 / dq if dq > 1e-6 else np.inf
                ruler2_state['dq'] = dq
                ruler2_state['d_spacing'] = d_spacing
                sync_target_dict()
                artists['ruler2_text'].set_text(f"Δq = {dq:.4f} 1/nm\nd = {d_spacing:.4f} nm")
                canvas.draw_idle()

    canvas.mpl_connect('button_press_event', onclick)
    canvas.mpl_connect('motion_notify_event', onmove)

    # Actions
    def open_config_dialog_action():
        nonlocal config
        dialog = ConfigDialog(root, config)
        root.wait_window(dialog)
        if dialog.updated:
            config = load_config()
            new_theme = config.get("theme", "dark")
            ctk.set_appearance_mode(new_theme)
            
            if new_theme == "dark":
                plt.style.use('dark_background')
            else:
                plt.style.use('default')

            style_matplotlib_toolbar(toolbar, new_theme)

            state['left_cmap'] = config["left_cmap"]
            state['right_cmap'] = config["right_cmap"]
            var_l_cmap.set(config["left_cmap"])
            var_r_cmap.set(config["right_cmap"])
            var_dx.set(str(config["dx"]))
            var_dy.set(str(config["dy"]))
            var_rin.set(str(config["r_in"]))
            var_rout.set(str(config["r_out"]))
            if state['myset']:
                rebuild_plot()

    def open_file_dialog_action():
        dialog = FileImportDialog(root, config)
        root.wait_window(dialog)

        if dialog.result:
            res = dialog.result
            state['path'] = res['path']
            state['filebasename'] = res['filebasename']
            state['filesuffix'] = res['filesuffix']
            state['format_type'] = res['format_type']

            config['bin_scan_y'], config['bin_scan_x'] = res['bin_scan']
            config['bin_det_y'], config['bin_det_x'] = res['bin_det']
            save_config(config)

            try:
                new_set = nbed.pyNBED()
                args = {
                    'bin_scan': res['bin_scan'],
                    'bin_det': res['bin_det']
                }
                if res['is_zipfile']:
                    args['is_zipfile'] = True

                new_set.LoadFile(res['full_path'], type=state['format_type'], **args)
                state['myset'] = new_set

                if os.path.exists(config["cal_db_path"]):
                    cal_mgr = nbed.MicroscopeCalibrationManager(config["cal_db_path"])
                    samp, qsamp = cal_mgr.get_calibration_from_metadata(
                        new_set.metadata, 
                        type=state['format_type']
                    )
                    state['re_samp'] = samp if samp else 1.0 * res['bin_scan'][1]
                    state['rec_samp'] = qsamp if qsamp else 1.0 * res['bin_det'][1]

                rebuild_plot()
                lbl_status.configure(
                    text=f"Loaded ({state['format_type']}): {res['filebasename']}{res['filesuffix']} "
                         f"[Calibrated: {state['re_samp']:.4g} nm/px, {state['rec_samp']:.4g} 1/nm/px]"
                )
            except Exception as e:
                messagebox.showerror("Loading Error", f"Failed to load file ({state['format_type']}):\n{e}")

    def open_sampling_dialog_action():
        if state['myset'] is None:
            messagebox.showwarning("No Dataset Loaded", "Please load a dataset before adjusting sampling values.")
            return

        dialog = ManualSamplingDialog(root, current_re_samp=state['re_samp'], current_rec_samp=state['rec_samp'])
        root.wait_window(dialog)

        if dialog.result:
            state['re_samp'] = dialog.result['re_samp']
            state['rec_samp'] = dialog.result['rec_samp']
            rebuild_plot()
            lbl_status.configure(
                text=f"Loaded ({state['format_type']}): {state['filebasename']}{state['filesuffix']} "
                     f"[Calibrated: {state['re_samp']:.4g} nm/px, {state['rec_samp']:.4g} 1/nm/px]"
            )

    def open_metadata_dialog_action():
        if state['myset'] is None:
            messagebox.showwarning("No Dataset Loaded", "Please load a dataset to view its metadata.")
            return

        dataset_label = f"{state['filebasename']}{state['filesuffix']}"
        dialog = MetadataViewerDialog(root, metadata_dict=state['myset'].metadata, dataset_name=dataset_label)
        root.wait_window(dialog)

    # Menu Bar
    menu_bar = tk.Menu(root)
    file_menu = tk.Menu(menu_bar, tearoff=0)
    file_menu.add_command(label="Open Dataset...", command=open_file_dialog_action)
    file_menu.add_command(label="Calibrate Sampling...", command=open_sampling_dialog_action)
    file_menu.add_command(label="View Metadata...", command=open_metadata_dialog_action)
    file_menu.add_separator()
    file_menu.add_command(label="Settings...", command=open_config_dialog_action)
    file_menu.add_command(label="About VirtualSTEMExplorerTK", command=show_about_dialog)
    file_menu.add_separator()
    file_menu.add_command(label="Exit", command=root.quit)
    menu_bar.add_cascade(label="File", menu=file_menu)
    root.configure(menu=menu_bar)

    # Control Dock Frame
    ctrl_frame = ctk.CTkFrame(root, corner_radius=0)
    ctrl_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=8)

    # 1. Left Contrast Controls
    f_left = ctk.CTkFrame(ctrl_frame)
    f_left.pack(side=tk.LEFT, fill=tk.Y, padx=4, pady=4)
    ctk.CTkLabel(f_left, text="Left Panel", font=("Arial", 11, "bold")).grid(row=0, column=0, columnspan=3, pady=(4,2))

    ctk.CTkLabel(f_left, text="Clip L/H:").grid(row=1, column=0, sticky='w', padx=4)
    var_l_clip_low = ctk.StringVar(value="0.0")
    var_l_clip_high = ctk.StringVar(value="100.0")
    ctk.CTkEntry(f_left, textvariable=var_l_clip_low, width=42).grid(row=1, column=1, padx=2)
    ctk.CTkEntry(f_left, textvariable=var_l_clip_high, width=42).grid(row=1, column=2, padx=2)

    ctk.CTkLabel(f_left, text="Gamma:").grid(row=2, column=0, sticky='w', padx=4)
    var_l_gamma = ctk.StringVar(value="1.0")
    ctk.CTkEntry(f_left, textvariable=var_l_gamma, width=42).grid(row=2, column=1, padx=2)
    ctk.CTkButton(f_left, text="Apply", width=42, command=update_left).grid(row=2, column=2, padx=2)

    ctk.CTkLabel(f_left, text="Cmap:").grid(row=3, column=0, sticky='w', padx=4, pady=2)
    var_l_cmap = ctk.StringVar(value=state['left_cmap'])
    cmap_opts_l = ['gray', 'gray_r', 'inferno', 'magma', 'viridis', 'plasma', 'turbo', 'cividis', 'hsv', 'twilight']
    
    def on_l_cmap_select(choice):
        state['left_cmap'] = choice
        config['left_cmap'] = choice
        save_config(config)
        if state['myset']:
            artists['im1'].set_cmap(choice)
            sync_target_dict()
            update_left()

    cb_l_cmap = ctk.CTkOptionMenu(f_left, variable=var_l_cmap, values=cmap_opts_l, width=90, command=on_l_cmap_select)
    cb_l_cmap.grid(row=3, column=1, columnspan=2, pady=4)

    # 2. Mask Mode (COMPACT RADIO BUTTON GRID WITH SMALL FONT)
    f_mode = ctk.CTkFrame(ctrl_frame)
    f_mode.pack(side=tk.LEFT, fill=tk.Y, padx=4, pady=4)
    ctk.CTkLabel(f_mode, text="Mask Mode", font=("Arial", 11, "bold")).grid(row=0, column=0, columnspan=2, pady=(4,2))

    var_mode = ctk.StringVar(value=state['calc_mode'])
    modes = ['Sum', 'Variance', 'Fluctuation', 'COM Mag', 'COM Azimuth']

    def on_mode_change():
        chosen_mode = var_mode.get()
        state['calc_mode'] = chosen_mode
        if chosen_mode == 'COM Azimuth':
            state['left_cmap'] = 'hsv'
            var_l_cmap.set('hsv')
            if state['myset']: artists['im1'].set_cmap('hsv')
        sync_target_dict()
        refresh_detector_mask()

    for i, m in enumerate(modes):
        rb = ctk.CTkRadioButton(
            f_mode, 
            text=m, 
            value=m, 
            variable=var_mode, 
            command=on_mode_change,
            font=("Arial", 10),
            radiobutton_width=14,
            radiobutton_height=14,
            border_width_unchecked=2,
            border_width_checked=5
        )
        rb.grid(row=(i % 3) + 1, column=i // 3, sticky='w', padx=6, pady=2)

    # 3. Right Contrast Controls
    f_right = ctk.CTkFrame(ctrl_frame)
    f_right.pack(side=tk.LEFT, fill=tk.Y, padx=4, pady=4)
    ctk.CTkLabel(f_right, text="Right Panel", font=("Arial", 11, "bold")).grid(row=0, column=0, columnspan=3, pady=(4,2))

    ctk.CTkLabel(f_right, text="Clip L/H:").grid(row=1, column=0, sticky='w', padx=4)
    var_r_clip_low = ctk.StringVar(value="0.0")
    var_r_clip_high = ctk.StringVar(value="100.0")
    ctk.CTkEntry(f_right, textvariable=var_r_clip_low, width=42).grid(row=1, column=1, padx=2)
    ctk.CTkEntry(f_right, textvariable=var_r_clip_high, width=42).grid(row=1, column=2, padx=2)

    ctk.CTkLabel(f_right, text="Gamma:").grid(row=2, column=0, sticky='w', padx=4)
    var_r_gamma = ctk.StringVar(value="1.0")
    ctk.CTkEntry(f_right, textvariable=var_r_gamma, width=42).grid(row=2, column=1, padx=2)
    ctk.CTkButton(f_right, text="Apply", width=42, command=update_right).grid(row=2, column=2, padx=2)

    ctk.CTkLabel(f_right, text="Cmap:").grid(row=3, column=0, sticky='w', padx=4, pady=2)
    var_r_cmap = ctk.StringVar(value=state['right_cmap'])
    cmap_opts_r = ['gray', 'gray_r', 'inferno', 'magma', 'viridis', 'plasma', 'turbo', 'cividis']

    def on_r_cmap_select(choice):
        state['right_cmap'] = choice
        config['right_cmap'] = choice
        save_config(config)
        if state['myset']:
            artists['im2'].set_cmap(choice)
            sync_target_dict()
            update_right()

    cb_r_cmap = ctk.CTkOptionMenu(f_right, variable=var_r_cmap, values=cmap_opts_r, width=90, command=on_r_cmap_select)
    cb_r_cmap.grid(row=3, column=1, columnspan=2, pady=4)

    # 4. Mask & ROI Size Settings
    f_params = ctk.CTkFrame(ctrl_frame)
    f_params.pack(side=tk.LEFT, fill=tk.Y, padx=4, pady=4)
    ctk.CTkLabel(f_params, text="ROI & Mask Radii", font=("Arial", 11, "bold")).grid(row=0, column=0, columnspan=5, pady=(4,2))

    ctk.CTkLabel(f_params, text="dx:").grid(row=1, column=0, sticky='e')
    var_dx = ctk.StringVar(value=str(state['dx']))
    ctk.CTkEntry(f_params, textvariable=var_dx, width=38).grid(row=1, column=1)

    ctk.CTkLabel(f_params, text="dy:").grid(row=1, column=2, sticky='e')
    var_dy = ctk.StringVar(value=str(state['dy']))
    ctk.CTkEntry(f_params, textvariable=var_dy, width=38).grid(row=1, column=3)

    def set_roi_cmd():
        config['dx'] = int(var_dx.get())
        config['dy'] = int(var_dy.get())
        save_config(config)
        refresh_spatial_roi()

    ctk.CTkButton(f_params, text="Set ROI", width=55, command=set_roi_cmd).grid(row=1, column=4, padx=4)

    ctk.CTkLabel(f_params, text="r_in:").grid(row=2, column=0, sticky='e')
    var_rin = ctk.StringVar(value=str(state['r_in']))
    ctk.CTkEntry(f_params, textvariable=var_rin, width=38).grid(row=2, column=1)

    ctk.CTkLabel(f_params, text="r_out:").grid(row=2, column=2, sticky='e')
    var_rout = ctk.StringVar(value=str(state['r_out']))
    ctk.CTkEntry(f_params, textvariable=var_rout, width=38).grid(row=2, column=3)

    def set_mask_cmd():
        config['r_in'] = float(var_rin.get())
        config['r_out'] = float(var_rout.get())
        save_config(config)
        refresh_detector_mask()

    ctk.CTkButton(f_params, text="Set Mask", width=55, command=set_mask_cmd).grid(row=2, column=4, padx=4)

    # 5. Measurement Tools & Publication Export
    f_actions = ctk.CTkFrame(ctrl_frame)
    f_actions.pack(side=tk.LEFT, fill=tk.Y, padx=4, pady=4)
    ctk.CTkLabel(f_actions, text="Tools & Export", font=("Arial", 11, "bold")).grid(row=0, column=0, columnspan=2, pady=(4,2))

    btn_r1 = ctk.CTkButton(f_actions, text="Ruler Real", width=90)
    btn_r2 = ctk.CTkButton(f_actions, text="Ruler Recip", width=90)

    def toggle_ruler1():
        if state['myset'] is None: return
        ruler1_state['active'] = not ruler1_state['active']
        ruler1_state['p1'] = None; ruler1_state['p2'] = None; ruler1_state['length'] = 0.0
        sync_target_dict()
        if ruler1_state['active']:
            btn_r1.configure(text="Real ON", fg_color="#c2185b")
            artists['ruler1_text'].set_visible(True)
            artists['ruler1_text'].set_text("Click Point A on Virtual Image...")
        else:
            btn_r1.configure(text="Ruler Real", fg_color=ctk.ThemeManager.theme["CTkButton"]["fg_color"])
            artists['ruler1_line'].set_data([], [])
            artists['ruler1_text'].set_visible(False)
        canvas.draw_idle()

    def toggle_ruler2():
        if state['myset'] is None: return
        ruler2_state['active'] = not ruler2_state['active']
        ruler2_state['p1'] = None; ruler2_state['p2'] = None
        ruler2_state['dq'] = 0.0; ruler2_state['d_spacing'] = 0.0
        sync_target_dict()
        if ruler2_state['active']:
            btn_r2.configure(text="Recip ON", fg_color="#00838f")
            artists['ruler2_text'].set_visible(True)
            artists['ruler2_text'].set_text("Click Point A on Diffraction Pattern...")
        else:
            btn_r2.configure(text="Ruler Recip", fg_color=ctk.ThemeManager.theme["CTkButton"]["fg_color"])
            artists['ruler2_line'].set_data([], [])
            artists['ruler2_text'].set_visible(False)
        canvas.draw_idle()

    btn_r1.configure(command=toggle_ruler1)
    btn_r2.configure(command=toggle_ruler2)
    btn_r1.grid(row=1, column=0, padx=2, pady=2)
    btn_r2.grid(row=1, column=1, padx=2, pady=2)

    # Beam Alignment Button
    btn_align = ctk.CTkButton(f_actions, text="🎯 Align Beam", width=184, fg_color="#2e7d32", hover_color="#1b5e20", command=align_diffraction_centers)
    btn_align.grid(row=2, column=0, columnspan=2, padx=2, pady=2)

    def save_figure_callback():
        if state['myset'] is None:
            messagebox.showwarning("No Data", "Please load a dataset before saving.")
            return
        next_idx, filename = get_next_filename()
        try:
            fig.savefig(filename, bbox_inches='tight', pad_inches=0.05)
            next_idx_new, _ = get_next_filename()
            btn_save.configure(text=f"Export Fig #{next_idx_new:03d}")
            messagebox.showinfo("Export Success", f"Successfully exported:\n{filename}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export figure:\n{e}")

    btn_save = ctk.CTkButton(f_actions, text="Export Fig #001", width=184, command=save_figure_callback)
    btn_save.grid(row=3, column=0, columnspan=2, padx=2, pady=2)

    root.mainloop()
    return target_dict


def main():
    VirtualSTEMExplorerCTK(initial_filepath=None)

if __name__ == "__main__":
    main()
    

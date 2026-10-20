import os
import sys
import glob
import re
import json
import time
from pathlib import Path
import numpy as np
from scipy.ndimage import median_filter

import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle, Circle, Polygon

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

# Default Ring Presets defined directly as reciprocal radii q (1/nm)
DEFAULT_RING_PRESETS = {
    "Ice (P63mm)": [2.571, 2.73, 3.75, 4.45, 5.13, 5.22],
    "Gold (Au)": [2.12, 2.45, 3.47, 4.06, 4.24, 4.90],
    "Silicon (Si)": [1.57, 2.56, 3.01, 3.62, 3.95, 4.44]
}

DEFAULT_CONFIG = {
    "theme": "dark",
    "cal_db_path": str(Path.home() / "MicroscopeCalibrationDB.json"),
    "left_cmap": "gray",
    "right_cmap": "turbo",
    "left_origin": "lower",
    "right_origin": "lower",
    "right_scale": "log",
    "dx": 5,
    "dy": 5,
    "r_in": 0.0,
    "r_out": 25.0,
    "bin_scan_x": 2,
    "bin_scan_y": 2,
    "bin_det_x": 1,
    "bin_det_y": 1,
    "bg_kernel": 15,
    "ring_markers_q": "2.571, 2.73, 3.75, 4.45, 5.13, 5.22",
    "ring_presets": DEFAULT_RING_PRESETS,
    "export_markers": True,
    "eps11": 1.0,
    "eps12": 0.0,
    "eps21": 0.0,
    "eps22": 1.0
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


def parse_ring_radii(str_val: str):
    radii = []
    if not str_val:
        return radii
    parts = str_val.replace("(", "").replace(")", "").replace("[", "").replace("]", "").split(",")
    for p in parts:
        p_clean = p.strip()
        if p_clean:
            try:
                q_radius = float(p_clean)
                if q_radius > 0:
                    radii.append(q_radius)
            except ValueError:
                pass
    return radii


# --- Modal Dialogs ---
class MetadataViewerDialog(ctk.CTkToplevel):
    def __init__(self, parent, metadata_dict, dataset_name="Dataset"):
        super().__init__(parent)
        self.title(f"Metadata Viewer - {dataset_name}")
        self.geometry("700x650")
        self.grab_set()

        self.metadata = metadata_dict if isinstance(metadata_dict, dict) else {}
        self.formatted_str = self._format_metadata(self.metadata)

        f_main = ctk.CTkFrame(self, corner_radius=10)
        f_main.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        f_top = ctk.CTkFrame(f_main, fg_color="transparent")
        f_top.pack(fill=tk.X, padx=10, pady=(10, 6))

        ctk.CTkLabel(f_top, text="Metadata Entries:", font=("Arial", 13, "bold")).pack(side=tk.LEFT)

        self.var_filter = ctk.StringVar()
        self.var_filter.trace_add("write", self._apply_filter)
        
        entry_filter = ctk.CTkEntry(f_top, placeholder_text="Filter keys...", textvariable=self.var_filter, width=220)
        entry_filter.pack(side=tk.RIGHT)

        self.txt_view = ctk.CTkTextbox(f_main, font=("Courier", 11), wrap="none")
        self.txt_view.pack(fill=tk.BOTH, expand=True, padx=10, pady=6)

        f_btn = ctk.CTkFrame(f_main, fg_color="transparent")
        f_btn.pack(fill=tk.X, padx=10, pady=(4, 10))

        btn_copy = ctk.CTkButton(f_btn, text="📋 Copy to Clipboard", command=self._copy_to_clipboard, width=150)
        btn_copy.pack(side=tk.LEFT)

        btn_close = ctk.CTkButton(f_btn, text="Close", command=self.destroy, fg_color="transparent", border_width=1, width=90)
        btn_close.pack(side=tk.RIGHT)

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

        filtered_lines = [line for line in self.formatted_str.splitlines() if query in line.lower()]
        self._display_text("\n".join(filtered_lines) if filtered_lines else f"No matches found for '{query}'.")

    def _copy_to_clipboard(self):
        text_to_copy = self.txt_view.get("1.0", tk.END).strip()
        self.clipboard_clear()
        self.clipboard_append(text_to_copy)
        messagebox.showinfo("Copied", "Metadata text copied to clipboard!", parent=self)


class ConfigDialog(ctk.CTkToplevel):
    def __init__(self, parent, current_cfg):
        super().__init__(parent)
        self.title("Preferences")
        self.geometry("520x720")
        self.resizable(False, False)
        self.grab_set()

        self.cfg = current_cfg.copy()
        self.updated = False

        f_main = ctk.CTkFrame(self, corner_radius=10)
        f_main.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        f_theme = ctk.CTkFrame(f_main)
        f_theme.pack(fill=tk.X, padx=10, pady=6)
        ctk.CTkLabel(f_theme, text="GUI Theme:", font=("Arial", 12, "bold")).pack(side=tk.LEFT, padx=10)
        self.var_theme = ctk.StringVar(value=self.cfg.get("theme", "dark"))
        cb_theme = ctk.CTkOptionMenu(f_theme, variable=self.var_theme, values=["dark", "light"], width=100)
        cb_theme.pack(side=tk.RIGHT, padx=10, pady=6)

        f_db = ctk.CTkFrame(f_main)
        f_db.pack(fill=tk.X, padx=10, pady=6)
        ctk.CTkLabel(f_db, text="Calibration DB Path:", font=("Arial", 12, "bold")).pack(anchor='w', padx=10, pady=(6,0))
        
        f_db_inner = ctk.CTkFrame(f_db, fg_color="transparent")
        f_db_inner.pack(fill=tk.X, padx=10, pady=6)
        self.var_db_path = ctk.StringVar(value=self.cfg.get("cal_db_path", ""))
        ctk.CTkEntry(f_db_inner, textvariable=self.var_db_path, width=320).pack(side=tk.LEFT, padx=(0, 6))
        ctk.CTkButton(f_db_inner, text="Browse...", width=80, command=self.browse_db).pack(side=tk.LEFT)

        f_disp = ctk.CTkFrame(f_main)
        f_disp.pack(fill=tk.X, padx=10, pady=6)
        cmap_opts = ['gray', 'gray_r', 'inferno', 'magma', 'viridis', 'plasma', 'turbo', 'cividis', 'hsv', 'twilight']
        origin_opts = ['lower', 'upper']

        row1 = ctk.CTkFrame(f_disp, fg_color="transparent")
        row1.pack(fill=tk.X, padx=10, pady=4)
        ctk.CTkLabel(row1, text="Left Cmap:").pack(side=tk.LEFT, padx=4)
        self.var_l_cmap = ctk.StringVar(value=self.cfg.get("left_cmap", "gray"))
        ctk.CTkOptionMenu(row1, variable=self.var_l_cmap, values=cmap_opts, width=110).pack(side=tk.LEFT, padx=4)

        ctk.CTkLabel(row1, text="Right Cmap:").pack(side=tk.LEFT, padx=(12, 4))
        self.var_r_cmap = ctk.StringVar(value=self.cfg.get("right_cmap", "turbo"))
        ctk.CTkOptionMenu(row1, variable=self.var_r_cmap, values=cmap_opts, width=110).pack(side=tk.LEFT, padx=4)

        row_ori = ctk.CTkFrame(f_disp, fg_color="transparent")
        row_ori.pack(fill=tk.X, padx=10, pady=4)
        ctk.CTkLabel(row_ori, text="Left Origin:").pack(side=tk.LEFT, padx=4)
        self.var_l_origin = ctk.StringVar(value=self.cfg.get("left_origin", "lower"))
        ctk.CTkOptionMenu(row_ori, variable=self.var_l_origin, values=origin_opts, width=110).pack(side=tk.LEFT, padx=4)

        ctk.CTkLabel(row_ori, text="Right Origin:").pack(side=tk.LEFT, padx=(12, 4))
        self.var_r_origin = ctk.StringVar(value=self.cfg.get("right_origin", "lower"))
        ctk.CTkOptionMenu(row_ori, variable=self.var_r_origin, values=origin_opts, width=110).pack(side=tk.LEFT, padx=4)

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

        f_presets = ctk.CTkFrame(f_main)
        f_presets.pack(fill=tk.X, padx=10, pady=6)
        ctk.CTkLabel(f_presets, text="Custom Ring Presets (JSON Dictionary q in 1/nm):", font=("Arial", 11, "bold")).pack(anchor='w', padx=10, pady=(6, 2))

        self.txt_presets = ctk.CTkTextbox(f_presets, height=110, font=("Courier", 11))
        self.txt_presets.pack(fill=tk.X, padx=10, pady=4)
        
        presets_dict = self.cfg.get("ring_presets", DEFAULT_RING_PRESETS)
        self.txt_presets.insert("1.0", json.dumps(presets_dict, indent=2))

        f_dist = ctk.CTkFrame(f_main)
        f_dist.pack(fill=tk.X, padx=10, pady=6)
        ctk.CTkLabel(f_dist, text="Reciprocal Distortion Matrix M:", font=("Arial", 12, "bold")).pack(anchor='w', padx=10, pady=(6, 2))

        m_grid = ctk.CTkFrame(f_dist, fg_color="transparent")
        m_grid.pack(padx=10, pady=4)

        self.var_eps11 = ctk.StringVar(value=str(self.cfg.get("eps11", 1.0)))
        self.var_eps12 = ctk.StringVar(value=str(self.cfg.get("eps12", 0.0)))
        self.var_eps21 = ctk.StringVar(value=str(self.cfg.get("eps21", 0.0)))
        self.var_eps22 = ctk.StringVar(value=str(self.cfg.get("eps22", 1.0)))

        ctk.CTkLabel(m_grid, text="ε11:").grid(row=0, column=0, padx=4, pady=2)
        ctk.CTkEntry(m_grid, textvariable=self.var_eps11, width=65).grid(row=0, column=1, padx=4, pady=2)

        ctk.CTkLabel(m_grid, text="ε12:").grid(row=0, column=2, padx=4, pady=2)
        ctk.CTkEntry(m_grid, textvariable=self.var_eps12, width=65).grid(row=0, column=3, padx=4, pady=2)

        ctk.CTkLabel(m_grid, text="ε21:").grid(row=1, column=0, padx=4, pady=2)
        ctk.CTkEntry(m_grid, textvariable=self.var_eps21, width=65).grid(row=1, column=1, padx=4, pady=2)

        ctk.CTkLabel(m_grid, text="ε22:").grid(row=1, column=2, padx=4, pady=2)
        ctk.CTkEntry(m_grid, textvariable=self.var_eps22, width=65).grid(row=1, column=3, padx=4, pady=2)

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
            raw_json = self.txt_presets.get("1.0", tk.END).strip()
            parsed_presets = json.loads(raw_json)
            if not isinstance(parsed_presets, dict):
                raise ValueError("Ring presets must be a JSON object/dictionary.")
            self.cfg["ring_presets"] = parsed_presets

            self.cfg["theme"] = self.var_theme.get()
            self.cfg["cal_db_path"] = self.var_db_path.get().strip()
            self.cfg["left_cmap"] = self.var_l_cmap.get()
            self.cfg["right_cmap"] = self.var_r_cmap.get()
            self.cfg["left_origin"] = self.var_l_origin.get()
            self.cfg["right_origin"] = self.var_r_origin.get()
            self.cfg["dx"] = int(self.var_dx.get())
            self.cfg["dy"] = int(self.var_dy.get())
            self.cfg["r_in"] = float(self.var_rin.get())
            self.cfg["r_out"] = float(self.var_rout.get())

            self.cfg["eps11"] = float(self.var_eps11.get())
            self.cfg["eps12"] = float(self.var_eps12.get())
            self.cfg["eps21"] = float(self.var_eps21.get())
            self.cfg["eps22"] = float(self.var_eps22.get())

            save_config(self.cfg)
            self.updated = True
            self.destroy()
        except Exception as e:
            messagebox.showerror("Validation Error", f"Failed to save settings:\n{e}", parent=self)


class FileImportDialog(ctk.CTkToplevel):
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
        
        self.fmt_menu = ctk.CTkOptionMenu(
            f_fmt, 
            variable=self.var_format, 
            values=["Auto-detect", "DECTRIS", "plain DECTRIS", "EMPAD", "PantaRhei"], 
            width=150,
            command=self._on_format_change
        )
        self.fmt_menu.pack(side=tk.LEFT)

        self.f_scan_shape = ctk.CTkFrame(f_main)
        ctk.CTkLabel(self.f_scan_shape, text="Manual Scan Shape (plain DECTRIS)", font=("Arial", 11, "bold")).pack(anchor='w', padx=10, pady=(6, 2))
        s_row = ctk.CTkFrame(self.f_scan_shape, fg_color="transparent")
        s_row.pack(fill=tk.X, padx=10, pady=4)
        ctk.CTkLabel(s_row, text="Scan Pixels (Y, X):").pack(side=tk.LEFT, padx=(0, 6))
        self.var_scany = ctk.StringVar(value="256")
        self.var_scanx = ctk.StringVar(value="256")
        ctk.CTkEntry(s_row, textvariable=self.var_scany, width=50).pack(side=tk.LEFT, padx=2)
        ctk.CTkEntry(s_row, textvariable=self.var_scanx, width=50).pack(side=tk.LEFT, padx=2)

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

    def _on_format_change(self, choice):
        if choice == "plain DECTRIS":
            self.f_scan_shape.pack(fill=tk.X, padx=10, pady=6, before=self.f_scan_shape.master.winfo_children()[-2])
        else:
            self.f_scan_shape.pack_forget()

    def auto_detect_format(self, filepath: str) -> str:
        fn_lower = filepath.lower()
        if fn_lower.endswith('.raw'):
            return 'EMPAD'
        elif fn_lower.endswith(('.prz', '.prz.zip')) or 'pantarhei' in fn_lower:
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
                ("All Supported Formats", "*.h5 *.hdf5 *.raw *.prz *.zip *.npy *.npz"),
                ("EMPAD Raw Files", "*.raw"),
                ("DECTRIS HDF5 Files", "*.h5 *.hdf5"),
                ("PantaRhei NumPy Archives", "*.prz *.zip *.npy *.npz"),
                ("All Files", "*.*")
            ]
        )
        if filename:
            self.var_filepath.set(filename)
            if self.var_format.get() == "Auto-detect":
                detected_fmt = self.auto_detect_format(filename)
                self.var_format.set(detected_fmt)
                self._on_format_change(detected_fmt)
            else:
                self._on_format_change(self.var_format.get())

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

        scan_shape = None
        if chosen_fmt == "plain DECTRIS":
            try:
                sy = int(self.var_scany.get())
                sx = int(self.var_scanx.get())
                if sy <= 0 or sx <= 0:
                    raise ValueError()
                scan_shape = (sy, sx)
            except ValueError:
                messagebox.showerror("Invalid Scan Shape", "Scan shape pixels must be positive integers.", parent=self)
                return

        fn_lower = filepath.lower()
        is_zip = fn_lower.endswith('.zip') or fn_lower.endswith('.prz.zip')

        filedir, filename = os.path.split(filepath)
        basename, suffix = os.path.splitext(filename)

        actual_type = "DECTRIS" if chosen_fmt in ["DECTRIS", "plain DECTRIS"] else chosen_fmt

        self.result = {
            "full_path": filepath,
            "path": filedir + "/",
            "filebasename": basename,
            "filesuffix": suffix,
            "format_type": actual_type,
            "is_plain_dectris": (chosen_fmt == "plain DECTRIS"),
            "scan_shape": scan_shape,
            "is_zipfile": is_zip,
            "bin_scan": (bs_y, bs_x),
            "bin_det": (bd_y, bd_x)
        }
        self.destroy()


class ManualSamplingDialog(ctk.CTkToplevel):
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


class DistortionCalibDialog(ctk.CTkToplevel):
    def __init__(self, parent, presets_dict, current_M, update_calib_rings_cb, close_calib_cb, app_config):
        super().__init__(parent)
        self.title("Reciprocal Distortion Matrix Calibration")
        self.geometry("520x560")
        self.resizable(False, False)
        
        self.presets = presets_dict
        self.update_calib_rings_cb = update_calib_rings_cb
        self.close_calib_cb = close_calib_cb
        self.app_config = app_config
        self.transferred = False

        self.base_M = current_M.copy()
        self.curr_M = current_M.copy()

        f_main = ctk.CTkFrame(self, corner_radius=10)
        f_main.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        ctk.CTkLabel(f_main, text="Distortion Matrix Calibration (±6% Range)", font=("Arial", 13, "bold")).pack(anchor='w', padx=10, pady=(6, 4))

        f_preset = ctk.CTkFrame(f_main)
        f_preset.pack(fill=tk.X, padx=10, pady=6)
        ctk.CTkLabel(f_preset, text="Preset Reference Rings:").pack(side=tk.LEFT, padx=10)
        
        preset_names = list(self.presets.keys())
        self.var_preset = ctk.StringVar(value=preset_names[0] if preset_names else "None")
        cb_preset = ctk.CTkOptionMenu(f_preset, variable=self.var_preset, values=preset_names, command=self.on_preset_change)
        cb_preset.pack(side=tk.RIGHT, padx=10, pady=6)

        f_sliders = ctk.CTkFrame(f_main)
        f_sliders.pack(fill=tk.X, padx=10, pady=6)

        self.sliders = {}
        self.labels = {}
        self.vars = {}

        keys = [('eps11', 0, 0), ('eps12', 0, 1), ('eps21', 1, 0), ('eps22', 1, 1)]
        
        for key, r, c in keys:
            base_val = self.base_M[r, c]
            
            if abs(base_val) > 1e-5:
                delta = abs(base_val) * 0.06
                min_v, max_v = base_val - delta, base_val + delta
            else:
                min_v, max_v = -0.06, 0.06

            f_s = ctk.CTkFrame(f_sliders, fg_color="transparent")
            f_s.pack(fill=tk.X, padx=8, pady=4)

            lbl = ctk.CTkLabel(f_s, text=f"ε{r+1}{c+1}: {base_val:.4f}", width=90, anchor='w')
            lbl.pack(side=tk.LEFT, padx=4)
            self.labels[key] = lbl

            var_s = tk.DoubleVar(value=base_val)
            self.vars[key] = var_s

            slider = ctk.CTkSlider(
                f_s, 
                from_=min_v, 
                to=max_v, 
                variable=var_s, 
                command=lambda val, k=key, row=r, col=c: self.on_slider_move(k, row, col, val)
            )
            slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
            self.sliders[key] = slider

            btn_reset = ctk.CTkButton(f_s, text="↺", width=28, command=lambda k=key, row=r, col=c, bv=base_val: self.reset_slider(k, row, col, bv))
            btn_reset.pack(side=tk.RIGHT, padx=2)

        f_out = ctk.CTkFrame(f_main)
        f_out.pack(fill=tk.X, padx=10, pady=8)
        ctk.CTkLabel(f_out, text="Calibrated Distortion Matrix M (Inverse):", font=("Arial", 11, "bold")).pack(anchor='w', padx=10, pady=(4,2))
        
        self.lbl_matrix = ctk.CTkLabel(f_out, text=self._format_matrix_str(), font=("Courier", 12))
        self.lbl_matrix.pack(padx=10, pady=4)

        f_btn = ctk.CTkFrame(f_main, fg_color="transparent")
        f_btn.pack(pady=10)

        ctk.CTkButton(f_btn, text="⚙️ Transfer to Settings", fg_color="#0288d1", hover_color="#01579b", command=self.transfer_to_settings, width=170).pack(side=tk.LEFT, padx=6)
        ctk.CTkButton(f_btn, text="Close", command=self.on_close, fg_color="transparent", border_width=1, width=90).pack(side=tk.LEFT, padx=6)

        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.on_preset_change(self.var_preset.get())

    def _format_matrix_str(self):
        return (f"[ {self.curr_M[0,0]:.5f} ,  {self.curr_M[0,1]:.5f} ]\n"
                f"[ {self.curr_M[1,0]:.5f} ,  {self.curr_M[1,1]:.5f} ]")

    def on_slider_move(self, key, row, col, value):
        val_float = float(value)
        self.curr_M[row, col] = val_float
        self.labels[key].configure(text=f"ε{row+1}{col+1}: {val_float:.4f}")
        self.lbl_matrix.configure(text=self._format_matrix_str())
        self.notify_update()

    def reset_slider(self, key, row, col, base_val):
        self.vars[key].set(base_val)
        self.on_slider_move(key, row, col, base_val)

    def on_preset_change(self, choice):
        self.notify_update()

    def notify_update(self):
        preset_key = self.var_preset.get()
        radii_q = self.presets.get(preset_key, [])
        self.update_calib_rings_cb(radii_q, self.curr_M)

    def transfer_to_settings(self):
        det = np.linalg.det(self.curr_M)
        if abs(det) > 1e-9:
            norm_M = self.curr_M / np.sqrt(abs(det))
        else:
            norm_M = self.curr_M

        self.app_config['eps11'] = float(norm_M[0, 0])
        self.app_config['eps12'] = float(norm_M[0, 1])
        self.app_config['eps21'] = float(norm_M[1, 0])
        self.app_config['eps22'] = float(norm_M[1, 1])
        
        save_config(self.app_config)
        self.transferred = True
        
        messagebox.showinfo(
            "Transferred & Normalized", 
            f"Normalized Distortion Matrix (det = 1.0) transferred to Settings:\n\n"
            f"[ {norm_M[0,0]:.5f} , {norm_M[0,1]:.5f} ]\n"
            f"[ {norm_M[1,0]:.5f} , {norm_M[1,1]:.5f} ]", 
            parent=self
        )

    def on_close(self):
        self.close_calib_cb()
        self.destroy()


# --- Main Application Explorer ---
def VirtualSTEMExplorerCTK(myset=None, initial_filepath=None, initial_format=None):
    config = load_config()
    target_dict = {}

    theme_mode = config.get("theme", "dark")
    ctk.set_appearance_mode(theme_mode)
    ctk.set_default_color_theme("blue")

    root = ctk.CTk()
    root.title("VirtualSTEMExplorerTK [CustomTkinter]")
    root.geometry("1720x960")

    if theme_mode == "dark":
        plt.style.use('dark_background')
    else:
        plt.style.use('default')

    ring_presets = config.get("ring_presets", DEFAULT_RING_PRESETS)

    state = {
        'myset': myset,
        'path': "",
        'filebasename': "VirtualSTEMExplorer_Export",
        'filesuffix': ".raw",
        'format_type': "EMPAD",
        're_samp': 1.0,
        'rec_samp': 1.0,
        'x': 0, 'y': 0,
        'scanx': 0, 'scany': 0,
        'detx': 0, 'dety': 0,
        'dx': config["dx"], 'dy': config["dy"],
        'cqx': 0, 'cqy': 0,
        'r_in': config["r_in"], 'r_out': config["r_out"],
        'calc_mode': 'Sum',
        'left_cmap': config["left_cmap"],
        'right_cmap': config["right_cmap"],
        'left_origin': config.get("left_origin", "lower"),
        'right_origin': config.get("right_origin", "lower"),
        'right_scale': config.get("right_scale", "log"),
        'eps11': config.get("eps11", 1.0),
        'eps12': config.get("eps12", 0.0),
        'eps21': config.get("eps21", 0.0),
        'eps22': config.get("eps22", 1.0),
        'ring_markers': [],
        'ring_presets': ring_presets,
        'bg_sub': False,
        'bg_kernel': config.get("bg_kernel", 15),
        'calib_active': False
    }

    pan_state = {'active': False, 'ax': None, 'start_x': 0, 'start_y': 0, 'xlim': None, 'ylim': None}
    click_tracker = {'last_time': 0, 'last_ax': None}

    def show_about_dialog():
        messagebox.showinfo(
            "About VirtualSTEMExplorerTK",
            "VirtualSTEMExplorerTK v2.3 (CustomTkinter)\n"
            "Interactive 4D-STEM Analysis Suite\n\n"
            "Copyright © 2026 L. Houben - Weizmann Institute of Science."
        )

    ruler1_state = {'active': False, 'p1': None, 'p2': None, 'length': 0.0}
    ruler2_state = {'active': False, 'p1': None, 'p2': None, 'dq': 0.0, 'd_spacing': 0.0}

    top_bar = ctk.CTkFrame(root, height=45, corner_radius=0)
    top_bar.pack(side=tk.TOP, fill=tk.X)

    btn_open = ctk.CTkButton(top_bar, text="📁 Open Dataset...", width=130, command=lambda: open_file_dialog_action())
    btn_open.pack(side=tk.LEFT, padx=8, pady=8)

    btn_config = ctk.CTkButton(top_bar, text="⚙️ Settings...", width=110, fg_color="transparent", border_width=1, command=lambda: open_config_dialog_action())
    btn_config.pack(side=tk.LEFT, padx=4, pady=8)

    def reset_left_view():
        if state['myset'] is None: return
        scanx, scany = state['scanx'], state['scany']
        ax1.set_xlim(0, scanx * state['re_samp'])
        if state['left_origin'] == 'upper':
            ax1.set_ylim(scany * state['re_samp'], 0)
        else:
            ax1.set_ylim(0, scany * state['re_samp'])
        canvas.draw_idle()

    def reset_right_view():
        if state['myset'] is None: return
        extent_right = get_reciprocal_extent(state['cqx'], state['cqy'])
        ax2.set_xlim(extent_right[0], extent_right[1])
        ax2.set_ylim(extent_right[2], extent_right[3])
        canvas.draw_idle()

    btn_reset_real = ctk.CTkButton(top_bar, text="↺ Reset Real", width=95, fg_color="#37474f", hover_color="#263238", command=reset_left_view)
    btn_reset_real.pack(side=tk.LEFT, padx=4, pady=8)

    btn_reset_recip = ctk.CTkButton(top_bar, text="↺ Reset Recip", width=95, fg_color="#37474f", hover_color="#263238", command=reset_right_view)
    btn_reset_recip.pack(side=tk.LEFT, padx=4, pady=8)

    lbl_status = ctk.CTkLabel(top_bar, text="No dataset loaded. Click 'Open Dataset' to start.", font=("Arial", 11, "italic"))
    lbl_status.pack(side=tk.LEFT, padx=12, pady=8)

    canvas_frame = ctk.CTkFrame(root)
    canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=(4, 0))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.subplots_adjust(top=0.88, bottom=0.15, left=0.08, right=0.92, wspace=0.35)

    canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    artists = {'ring_patches': [], 'calib_patches': []}

    def get_reciprocal_extent(center_x, center_y):
        x_min = -center_x * state['rec_samp']
        x_max = (state['detx'] - center_x) * state['rec_samp']
        
        if state['right_origin'] == 'upper':
            y_min = -(state['dety'] - center_y) * state['rec_samp']
            y_max = center_y * state['rec_samp']
        else:
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
        target_dict['left_origin'] = state['left_origin']
        target_dict['right_origin'] = state['right_origin']
        target_dict['right_scale'] = state['right_scale']
        target_dict['eps11'] = float(state['eps11'])
        target_dict['eps12'] = float(state['eps12'])
        target_dict['eps21'] = float(state['eps21'])
        target_dict['eps22'] = float(state['eps22'])
        target_dict['ring_markers'] = list(state['ring_markers'])
        target_dict['bg_sub'] = bool(state['bg_sub'])
        target_dict['bg_kernel'] = int(state['bg_kernel'])
        target_dict['mode'] = state['calc_mode']
        
        scan_x = state.get('scanx', 0)
        target_dict['index'] = int(state['y'] * scan_x + state['x']) if scan_x > 0 else 0
        
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
        
        avg_slice = np.mean(state['myset'].data[y_min:y_max, x_min:x_max, :, :], axis=(0, 1)).astype(np.float64)

        if state['bg_sub']:
            det_x = state['myset'].dim[3]
            max_k = max(2, det_x // 4)
            ksize = int(np.clip(state['bg_kernel'], 2, max_k))
            if ksize % 2 == 0:
                ksize += 1
            bg = median_filter(avg_slice, size=ksize)
            avg_slice = np.maximum(0.0, avg_slice - bg)

        if state['right_scale'] == 'log':
            return np.log(avg_slice + 1.0)
        else:
            return avg_slice

    def apply_contrast(data, clip_percentiles, gamma):
        p_low, p_high = clip_percentiles
        vmin, vmax = np.percentile(data, [p_low, p_high])
        if vmax == vmin:
            vmax = vmin + 1e-5
        norm = np.clip((data - vmin) / (vmax - vmin), 0.0, 1.0)
        return np.power(norm, gamma)

    def redraw_ring_markers():
        for p in artists['ring_patches']:
            try: p.remove()
            except Exception: pass
        artists['ring_patches'].clear()

        if state['myset'] is None or state['calib_active']: return

        for q_radius in state['ring_markers']:
            circ = Circle((0, 0), q_radius, color='cyan', fill=False, linestyle=':', linewidth=1.2, alpha=0.85)
            ax2.add_patch(circ)
            artists['ring_patches'].append(circ)

    def draw_calib_elliptical_rings(radii_q, test_M):
        for p in artists['calib_patches']:
            try: p.remove()
            except Exception: pass
        artists['calib_patches'].clear()

        if state['myset'] is None: return

        try:
            inv_M = np.linalg.inv(test_M)
        except np.linalg.LinAlgError:
            inv_M = np.eye(2)

        theta = np.linspace(0, 2*np.pi, 120)
        
        for q_r in radii_q:
            circle_pts = np.vstack([q_r * np.cos(theta), q_r * np.sin(theta)])
            distorted_pts = inv_M @ circle_pts
            poly_xy = distorted_pts.T

            poly = Polygon(poly_xy, closed=True, edgecolor='#ff9800', facecolor='none', linestyle='--', linewidth=1.4)
            ax2.add_patch(poly)
            artists['calib_patches'].append(poly)

        canvas.draw_idle()

    def clear_calib_rings():
        for p in artists['calib_patches']:
            try: p.remove()
            except Exception: pass
        artists['calib_patches'].clear()
        state['calib_active'] = False
        redraw_ring_markers()
        canvas.draw_idle()

    def apply_ring_preset_by_key(preset_key: str):
        presets = state['ring_presets']
        if preset_key in presets:
            q_radii = presets[preset_key]
            state['ring_markers'] = list(q_radii)
            if q_radii:
                var_ring_q.set(str(q_radii[0]))
            sync_target_dict()
            redraw_ring_markers()
            canvas.draw_idle()

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
        if state['left_origin'] == 'upper':
            y_box_min = (state['scany'] - 1 - state['y'] - state['dy'] - 0.5) * state['re_samp']
        else:
            y_box_min = (state['y'] - state['dy'] - 0.5) * state['re_samp']

        artists['roi_rect'].set_bounds((state['x'] - state['dx'] - 0.5) * state['re_samp'],
                                       y_box_min,
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
        artists['im1'] = ax1.imshow(img1_proc, cmap=state['left_cmap'], origin=state['left_origin'], vmin=0, vmax=1, extent=extent_left)
        ax1.set_title(f"Virtual Image [{state['calc_mode']}]", pad=15)
        ax1.set_xlabel("x (nm)"); ax1.set_ylabel("y (nm)")

        sec_ax1_x = ax1.secondary_xaxis('top', functions=(lambda val: val / state['re_samp'], lambda val: val * state['re_samp']))
        sec_ax1_y = ax1.secondary_yaxis('right', functions=(lambda val: val / state['re_samp'], lambda val: val * state['re_samp']))
        sec_ax1_x.set_xlabel("x (pixels)", labelpad=4)
        sec_ax1_y.set_ylabel("y (pixels)", labelpad=4)

        px_x = state['x'] * state['re_samp']
        px_y = (state['scany'] - 1 - state['y']) * state['re_samp'] if state['left_origin'] == 'upper' else state['y'] * state['re_samp']
        
        artists['v_line'] = ax1.axvline(px_x, color='yellow', lw=1)
        artists['h_line'] = ax1.axhline(px_y, color='yellow', lw=1)

        y_box_min = (state['scany'] - 1 - state['y'] - state['dy'] - 0.5) * state['re_samp'] if state['left_origin'] == 'upper' else (state['y'] - state['dy'] - 0.5) * state['re_samp']
        
        artists['roi_rect'] = Rectangle(((state['x'] - state['dx'] - 0.5) * state['re_samp'], y_box_min),
                                         (2*state['dx'] + 1) * state['re_samp'],
                                         (2*state['dy'] + 1) * state['re_samp'],
                                         linewidth=1, edgecolor='yellow', facecolor='none', linestyle='--')
        ax1.add_patch(artists['roi_rect'])

        # RIGHT Panel
        artists['im2'] = ax2.imshow(img2_proc, cmap=state['right_cmap'], origin=state['right_origin'], vmin=0, vmax=1, extent=extent_right)
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

        redraw_ring_markers()

        next_idx, _ = get_next_filename()
        btn_save.configure(text=f"Export Fig #{next_idx:03d}")
        sync_target_dict()
        canvas.draw()

    def align_diffraction_centers():
        if state['myset'] is None:
            messagebox.showwarning("No Dataset Loaded", "Please load a dataset before running center alignment.")
            return

        print("\n--- Starting Diffraction Pattern Alignment ---")
        data_4d = state['myset'].data
        sy, sx, dy, dx = data_4d.shape

        print("1. Computing peak positions across all diffraction frames...")
        flat_data = data_4d.reshape(sy, sx, -1)
        max_flat_idx = np.argmax(flat_data, axis=-1)
        cy_map, cx_map = np.unravel_index(max_flat_idx, (dy, dx))

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

    # Dynamic Right-Click Context Menu
    ctx_menu = tk.Menu(root, tearoff=0)
    
    def rebuild_context_menu():
        ctx_menu.delete(0, tk.END)
        
        preset_sub = tk.Menu(ctx_menu, tearoff=0)
        for key in state['ring_presets'].keys():
            preset_sub.add_command(
                label=key, 
                command=lambda k=key: apply_ring_preset_by_key(k)
            )
        
        ctx_menu.add_cascade(label="🧊 Add Ring Preset", menu=preset_sub)
        ctx_menu.add_separator()
        ctx_menu.add_command(label="🗑️ Clear Ring Markers", command=lambda: clear_ring_markers_cmd())

    def show_context_menu(event):
        if event.inaxes == ax2:
            rebuild_context_menu()
            try:
                ctx_menu.tk_popup(event.guiEvent.x_root, event.guiEvent.y_root)
            finally:
                ctx_menu.grab_release()

    # --- Mouse Handlers & Custom Interactive Navigation ---
    def on_press(event):
        if state['myset'] is None or event.inaxes is None: return

        if event.button == 3:
            show_context_menu(event)
            return

        if event.button == 1:
            curr_time = time.time()
            dt = curr_time - click_tracker['last_time']
            is_double_click = (dt < 0.35) and (click_tracker['last_ax'] == event.inaxes)
            is_shift_click = (event.key == 'shift')
            
            click_tracker['last_time'] = curr_time
            click_tracker['last_ax'] = event.inaxes

            # 1. Measurement Rulers ONLY on SHIFT + CLICK
            if is_shift_click:
                if event.inaxes == ax1 and ruler1_state['active']:
                    pt = (event.xdata, event.ydata)
                    if ruler1_state['p1'] is None or ruler1_state['p2'] is not None:
                        ruler1_state['p1'] = pt
                        ruler1_state['p2'] = None
                        artists['ruler1_line'].set_data([pt[0]], [pt[1]])
                        artists['ruler1_text'].set_text(f"P1: ({pt[0]:.2f}, {pt[1]:.2f})\nShift-click P2...")
                    else:
                        ruler1_state['p2'] = pt
                        p1 = ruler1_state['p1']
                        artists['ruler1_line'].set_data([p1[0], pt[0]], [p1[1], pt[1]])
                        dr = np.hypot(pt[0] - p1[0], pt[1] - p1[1])
                        ruler1_state['length'] = dr
                        sync_target_dict()
                        artists['ruler1_text'].set_text(f"Δr = {dr:.3f} nm\n({dr/state['re_samp']:.1f} px)")
                        
                        # Auto-deactivate ruler once Point B is set
                        ruler1_state['active'] = False
                        btn_r1.configure(text="Ruler Real", fg_color=ctk.ThemeManager.theme["CTkButton"]["fg_color"])
                    canvas.draw_idle()
                    return

                elif event.inaxes == ax2 and ruler2_state['active']:
                    pt = (event.xdata, event.ydata)
                    if ruler2_state['p1'] is None or ruler2_state['p2'] is not None:
                        ruler2_state['p1'] = pt
                        ruler2_state['p2'] = None
                        artists['ruler2_line'].set_data([pt[0]], [pt[1]])
                        artists['ruler2_text'].set_text(f"P1: ({pt[0]:.2f}, {pt[1]:.2f})\nShift-click P2...")
                    else:
                        ruler2_state['p2'] = pt
                        p1 = ruler2_state['p1']
                        artists['ruler2_line'].set_data([p1[0], pt[0]], [p1[1], pt[1]])
                        
                        raw_dq = np.array([pt[0] - p1[0], pt[1] - p1[1]], dtype=float)
                        M = np.array([
                            [state['eps11'], state['eps12']],
                            [state['eps21'], state['eps22']]
                        ], dtype=float)
                        
                        corrected_dq = M @ raw_dq
                        dq = float(np.linalg.norm(corrected_dq))
                        d_spacing = 1.0 / dq if dq > 1e-6 else np.inf
                        
                        ruler2_state['dq'] = dq
                        ruler2_state['d_spacing'] = d_spacing
                        sync_target_dict()
                        artists['ruler2_text'].set_text(f"Δq = {dq:.4f} 1/nm\nd = {d_spacing:.4f} nm")
                        
                        # Auto-deactivate ruler once Point B is set
                        ruler2_state['active'] = False
                        btn_r2.configure(text="Ruler Recip", fg_color=ctk.ThemeManager.theme["CTkButton"]["fg_color"])
                    canvas.draw_idle()
                    return

            # 2. Regular Single-Click: Initialize Panning
            pan_state['active'] = True
            pan_state['ax'] = event.inaxes
            pan_state['start_x'] = event.x
            pan_state['start_y'] = event.y
            pan_state['xlim'] = event.inaxes.get_xlim()
            pan_state['ylim'] = event.inaxes.get_ylim()

            # 3. Double-Click Handler: Position Mask or ROI Crosshair
            if is_double_click:
                if event.inaxes == ax1:
                    click_x = int(np.clip(event.xdata / state['re_samp'], 0, state['scanx'] - 1))
                    raw_click_y = event.ydata / state['re_samp']
                    if state['left_origin'] == 'upper':
                        click_y = int(np.clip(state['scany'] - 1 - raw_click_y, 0, state['scany'] - 1))
                    else:
                        click_y = int(np.clip(raw_click_y, 0, state['scany'] - 1))

                    state['x'], state['y'] = click_x, click_y
                    px_x = state['x'] * state['re_samp']
                    px_y = (state['scany'] - 1 - state['y']) * state['re_samp'] if state['left_origin'] == 'upper' else state['y'] * state['re_samp']

                    artists['v_line'].set_xdata([px_x, px_x])
                    artists['h_line'].set_ydata([px_y, px_y])
                    refresh_spatial_roi()

                elif event.inaxes == ax2:
                    delta_qx_px = event.xdata / state['rec_samp']
                    delta_qy_px = event.ydata / state['rec_samp']

                    click_qx = state['cqx'] + delta_qx_px

                    if state['right_origin'] == 'upper':
                        click_qy = state['cqy'] - delta_qy_px
                    else:
                        click_qy = state['cqy'] + delta_qy_px

                    state['cqx'] = float(np.clip(click_qx, 0, state['detx'] - 1))
                    state['cqy'] = float(np.clip(click_qy, 0, state['dety'] - 1))
                    refresh_detector_mask()

    def on_release(event):
        pan_state['active'] = False
        pan_state['ax'] = None

    def on_motion(event):
        if state['myset'] is None: return

        if pan_state['active'] and pan_state['ax'] == event.inaxes and event.x is not None and event.y is not None:
            ax = pan_state['ax']
            dx_pixels = event.x - pan_state['start_x']
            dy_pixels = event.y - pan_state['start_y']

            bbox = ax.get_window_extent()
            width_px, height_px = bbox.width, bbox.height

            xlim = pan_state['xlim']
            ylim = pan_state['ylim']

            dx_data = -dx_pixels * (xlim[1] - xlim[0]) / width_px
            dy_data = -dy_pixels * (ylim[1] - ylim[0]) / height_px

            ax.set_xlim(xlim[0] + dx_data, xlim[1] + dx_data)
            ax.set_ylim(ylim[0] + dy_data, ylim[1] + dy_data)
            canvas.draw_idle()
            return

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
                
                raw_dq = np.array([p2[0] - p1[0], p2[1] - p1[1]], dtype=float)
                M = np.array([
                    [state['eps11'], state['eps12']],
                    [state['eps21'], state['eps22']]
                ], dtype=float)
                
                corrected_dq = M @ raw_dq
                dq = float(np.linalg.norm(corrected_dq))
                d_spacing = 1.0 / dq if dq > 1e-6 else np.inf
                
                ruler2_state['dq'] = dq
                ruler2_state['d_spacing'] = d_spacing
                sync_target_dict()
                artists['ruler2_text'].set_text(f"Δq = {dq:.4f} 1/nm\nd = {d_spacing:.4f} nm")
                canvas.draw_idle()

    def on_scroll(event):
        if state['myset'] is None or event.inaxes is None: return
        
        ax = event.inaxes
        scale_factor = 0.85 if event.button == 'up' else 1.15

        xdata, ydata = event.xdata, event.ydata
        if xdata is None or ydata is None: return

        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        new_width = (xlim[1] - xlim[0]) * scale_factor
        new_height = (ylim[1] - ylim[0]) * scale_factor

        rel_x = (xdata - xlim[0]) / (xlim[1] - xlim[0])
        rel_y = (ydata - ylim[0]) / (ylim[1] - ylim[0])

        ax.set_xlim([xdata - rel_x * new_width, xdata + (1.0 - rel_x) * new_width])
        ax.set_ylim([ydata - rel_y * new_height, ydata + (1.0 - rel_y) * new_height])
        canvas.draw_idle()

    canvas.mpl_connect('button_press_event', on_press)
    canvas.mpl_connect('button_release_event', on_release)
    canvas.mpl_connect('motion_notify_event', on_motion)
    canvas.mpl_connect('scroll_event', on_scroll)

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

            state['left_cmap'] = config["left_cmap"]
            state['right_cmap'] = config["right_cmap"]
            state['left_origin'] = config.get("left_origin", "lower")
            state['right_origin'] = config.get("right_origin", "lower")
            state['right_scale'] = config.get("right_scale", "log")
            
            state['eps11'] = config.get("eps11", 1.0)
            state['eps12'] = config.get("eps12", 0.0)
            state['eps21'] = config.get("eps21", 0.0)
            state['eps22'] = config.get("eps22", 1.0)

            state['bg_kernel'] = config.get("bg_kernel", 15)
            var_bg_kernel.set(str(state['bg_kernel']))
            var_r_scale.set(state['right_scale'].capitalize())

            state['ring_presets'] = config.get("ring_presets", DEFAULT_RING_PRESETS)
            preset_names = list(state['ring_presets'].keys())
            if preset_names:
                cb_preset_menu.configure(values=preset_names)
                var_preset_sel.set(preset_names[0])

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
                    'bin_det': res['bin_det'],
                    'scan_shape': res['scan_shape']
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

    def open_distortion_calib_dialog_action():
        if state['myset'] is None:
            messagebox.showwarning("No Dataset Loaded", "Please load a dataset before calibrating distortion.")
            return

        current_M = np.array([
            [state['eps11'], state['eps12']],
            [state['eps21'], state['eps22']]
        ], dtype=float)

        state['calib_active'] = True
        redraw_ring_markers()

        dialog = DistortionCalibDialog(
            root, 
            presets_dict=state['ring_presets'], 
            current_M=current_M, 
            update_calib_rings_cb=draw_calib_elliptical_rings,
            close_calib_cb=clear_calib_rings,
            app_config=config
        )
        root.wait_window(dialog)

        if dialog.transferred:
            cfg = load_config()
            state['eps11'] = cfg['eps11']
            state['eps12'] = cfg['eps12']
            state['eps21'] = cfg['eps21']
            state['eps22'] = cfg['eps22']
            sync_target_dict()

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

    # 2. Mask Mode
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

    # 3. Right Contrast & Scale Controls
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

    ctk.CTkLabel(f_right, text="Scale:").grid(row=3, column=0, sticky='w', padx=4, pady=2)
    var_r_scale = ctk.StringVar(value=state['right_scale'].capitalize())

    def on_r_scale_select(choice):
        chosen_scale = choice.lower()
        state['right_scale'] = chosen_scale
        config['right_scale'] = chosen_scale
        save_config(config)
        refresh_spatial_roi()

    cb_r_scale = ctk.CTkOptionMenu(f_right, variable=var_r_scale, values=["Log", "Linear"], width=90, command=on_r_scale_select)
    cb_r_scale.grid(row=3, column=1, columnspan=2, pady=2)

    ctk.CTkLabel(f_right, text="Cmap:").grid(row=4, column=0, sticky='w', padx=4, pady=2)
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
    cb_r_cmap.grid(row=4, column=1, columnspan=2, pady=2)

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

    # 5. Measurement Tools
    f_tools = ctk.CTkFrame(ctrl_frame)
    f_tools.pack(side=tk.LEFT, fill=tk.Y, padx=4, pady=4)
    ctk.CTkLabel(f_tools, text="Measurement Tools", font=("Arial", 11, "bold")).grid(row=0, column=0, columnspan=3, pady=(4,2))

    btn_r1 = ctk.CTkButton(f_tools, text="Ruler Real", width=75)
    btn_r2 = ctk.CTkButton(f_tools, text="Ruler Recip", width=75)
    btn_calib = ctk.CTkButton(f_tools, text="📐 Calib", width=60, fg_color="#f57c00", hover_color="#e65100", command=open_distortion_calib_dialog_action)

    def toggle_ruler1():
        if state['myset'] is None: return
        ruler1_state['active'] = not ruler1_state['active']
        ruler1_state['p1'] = None; ruler1_state['p2'] = None; ruler1_state['length'] = 0.0
        sync_target_dict()
        if ruler1_state['active']:
            btn_r1.configure(text="Real ON", fg_color="#c2185b")
            artists['ruler1_text'].set_visible(True)
            artists['ruler1_text'].set_text("Shift + Click Point A on Virtual Image...")
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
            artists['ruler2_text'].set_text("Shift + Click Point A on Diffraction Pattern...")
        else:
            btn_r2.configure(text="Ruler Recip", fg_color=ctk.ThemeManager.theme["CTkButton"]["fg_color"])
            artists['ruler2_line'].set_data([], [])
            artists['ruler2_text'].set_visible(False)
        canvas.draw_idle()

    btn_r1.configure(command=toggle_ruler1)
    btn_r2.configure(command=toggle_ruler2)
    btn_r1.grid(row=1, column=0, padx=2, pady=2)
    btn_r2.grid(row=1, column=1, padx=2, pady=2)
    btn_calib.grid(row=1, column=2, padx=2, pady=2)

    # Reciprocal Ring Overlay Controls
    f_rings = ctk.CTkFrame(f_tools, fg_color="transparent")
    f_rings.grid(row=2, column=0, columnspan=3, pady=(4, 2))

    ctk.CTkLabel(f_rings, text="Ring q (1/nm):").pack(side=tk.LEFT, padx=2)
    var_ring_q = ctk.StringVar(value="2.571")
    ctk.CTkEntry(f_rings, textvariable=var_ring_q, width=52).pack(side=tk.LEFT, padx=2)

    def add_ring_marker_cmd():
        try:
            val_q = float(var_ring_q.get())
            if val_q <= 0: raise ValueError()
            state['ring_markers'].append(val_q)
            sync_target_dict()
            redraw_ring_markers()
            canvas.draw_idle()
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a positive numeric q radius value in 1/nm.", parent=root)

    def clear_ring_markers_cmd():
        state['ring_markers'].clear()
        sync_target_dict()
        redraw_ring_markers()
        canvas.draw_idle()

    preset_keys = list(state['ring_presets'].keys())
    var_preset_sel = ctk.StringVar(value=preset_keys[0] if preset_keys else "Presets")

    def on_preset_dropdown_select(choice):
        apply_ring_preset_by_key(choice)

    cb_preset_menu = ctk.CTkOptionMenu(
        f_rings, 
        variable=var_preset_sel, 
        values=preset_keys if preset_keys else ["None"], 
        width=110, 
        command=on_preset_dropdown_select,
        font=("Arial", 10)
    )

    ctk.CTkButton(f_rings, text="➕ Add", width=42, command=add_ring_marker_cmd).pack(side=tk.LEFT, padx=2)
    cb_preset_menu.pack(side=tk.LEFT, padx=2)
    ctk.CTkButton(f_rings, text="🗑️ Clear", width=52, fg_color="#d32f2f", hover_color="#9a0007", command=clear_ring_markers_cmd).pack(side=tk.LEFT, padx=2)

    # 6. Dedicated Processing Menu
    f_proc = ctk.CTkFrame(ctrl_frame)
    f_proc.pack(side=tk.LEFT, fill=tk.Y, padx=4, pady=4)
    ctk.CTkLabel(f_proc, text="Processing", font=("Arial", 11, "bold")).grid(row=0, column=0, columnspan=3, pady=(4,2))

    btn_align = ctk.CTkButton(f_proc, text="🎯 Align Beam", width=120, fg_color="#2e7d32", hover_color="#1b5e20", command=align_diffraction_centers)
    btn_align.grid(row=1, column=0, columnspan=3, padx=4, pady=2)

    # Display-Only Background Subtraction Controls
    var_bg_sub = tk.BooleanVar(value=state['bg_sub'])
    var_bg_kernel = ctk.StringVar(value=str(state['bg_kernel']))

    def on_bg_sub_toggle():
        state['bg_sub'] = var_bg_sub.get()
        try:
            k = int(var_bg_kernel.get())
            det_x = state['myset'].dim[3] if state['myset'] is not None else 128
            max_k = max(2, det_x // 4)
            k = int(np.clip(k, 2, max_k))
            state['bg_kernel'] = k
            var_bg_kernel.set(str(k))
        except ValueError:
            state['bg_kernel'] = 15
        sync_target_dict()
        refresh_spatial_roi()

    chk_bg = ctk.CTkCheckBox(f_proc, text="Sub BG", variable=var_bg_sub, command=on_bg_sub_toggle, font=("Arial", 10), width=65)
    chk_bg.grid(row=2, column=0, padx=2, pady=2, sticky='w')

    ctk.CTkLabel(f_proc, text="k:").grid(row=2, column=1, sticky='e', padx=1)
    entry_kernel = ctk.CTkEntry(f_proc, textvariable=var_bg_kernel, width=38)
    entry_kernel.grid(row=2, column=2, padx=2, pady=2, sticky='w')

    def apply_proc_param_change(*args):
        if state['bg_sub']: on_bg_sub_toggle()

    var_bg_kernel.trace_add("write", apply_proc_param_change)

    # 7. Publication Export Menu
    f_export = ctk.CTkFrame(ctrl_frame)
    f_export.pack(side=tk.LEFT, fill=tk.Y, padx=4, pady=4)
    ctk.CTkLabel(f_export, text="Export Menu", font=("Arial", 11, "bold")).pack(pady=(4, 2))

    var_export_markers = tk.BooleanVar(value=config.get("export_markers", True))

    def on_export_markers_toggle():
        config['export_markers'] = var_export_markers.get()
        save_config(config)

    chk_exp_markers = ctk.CTkCheckBox(
        f_export, 
        text="Include Markers", 
        variable=var_export_markers, 
        command=on_export_markers_toggle,
        font=("Arial", 10)
    )
    chk_exp_markers.pack(padx=6, pady=(2, 4))

    def save_figure_callback():
        if state['myset'] is None:
            messagebox.showwarning("No Data", "Please load a dataset before saving.")
            return

        next_idx, filename = get_next_filename()
        include_markers = var_export_markers.get()

        overlay_artists = []
        if 'v_line' in artists: overlay_artists.append(artists['v_line'])
        if 'h_line' in artists: overlay_artists.append(artists['h_line'])
        if 'roi_rect' in artists: overlay_artists.append(artists['roi_rect'])
        if 'center_marker' in artists: overlay_artists.append(artists['center_marker'])
        if 'circle_in' in artists: overlay_artists.append(artists['circle_in'])
        if 'circle_out' in artists: overlay_artists.append(artists['circle_out'])
        if 'ruler1_line' in artists: overlay_artists.append(artists['ruler1_line'])
        if 'ruler1_text' in artists and ruler1_state['active']: overlay_artists.append(artists['ruler1_text'])
        if 'ruler2_line' in artists: overlay_artists.append(artists['ruler2_line'])
        if 'ruler2_text' in artists and ruler2_state['active']: overlay_artists.append(artists['ruler2_text'])
        overlay_artists.extend(artists.get('ring_patches', []))
        overlay_artists.extend(artists.get('calib_patches', []))

        saved_visibilities = [art.get_visible() for art in overlay_artists]

        try:
            if not include_markers:
                for art in overlay_artists:
                    art.set_visible(False)
                canvas.draw()

            fig.savefig(filename, bbox_inches='tight', pad_inches=0.05)

        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export figure:\n{e}")
            return
        finally:
            for art, vis in zip(overlay_artists, saved_visibilities):
                art.set_visible(vis)
            canvas.draw_idle()

        next_idx_new, _ = get_next_filename()
        btn_save.configure(text=f"Export Fig #{next_idx_new:03d}")
        messagebox.showinfo("Export Success", f"Successfully exported:\n{filename}\n(Include Markers: {include_markers})")

    btn_save = ctk.CTkButton(f_export, text="Export Fig #001", width=120, command=save_figure_callback)
    btn_save.pack(padx=8, pady=(2, 8))

    root.mainloop()
    return target_dict


def main():
    VirtualSTEMExplorerCTK(initial_filepath=None)

if __name__ == "__main__":
    main()
    

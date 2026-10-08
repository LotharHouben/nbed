import os
import sys
import glob
import re
import json
from pathlib import Path
import numpy as np

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
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
    return config_dir / "VirtualSTEMExplorerTK_config.json"

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


# --- Theme Styling Utility ---
def apply_theme_styles(root, theme_mode="dark"):
    """Applies desktop theme colors to Tkinter ttk widgets."""
    style = ttk.Style(root)
    style.theme_use('clam')

    if theme_mode == "dark":
        bg_dark = "#1e1e1e"
        bg_panel = "#2d2d2d"
        fg_text = "#ffffff"
        accent_btn = "#3c3c3c"
        active_btn = "#505050"

        root.configure(bg=bg_dark)
        
        style.configure(".", background=bg_panel, foreground=fg_text, bordercolor="#444444")
        style.configure("TFrame", background=bg_dark)
        style.configure("TLabelframe", background=bg_panel, foreground=fg_text, bordercolor="#555555")
        style.configure("TLabelframe.Label", background=bg_panel, foreground=fg_text)
        style.configure("TLabel", background=bg_panel, foreground=fg_text)
        style.configure("TButton", background=accent_btn, foreground=fg_text, bordercolor="#555555", padding=3)
        style.map("TButton", background=[("active", active_btn)])
        style.configure("TRadiobutton", background=bg_panel, foreground=fg_text)
        style.map("TRadiobutton", background=[("active", bg_panel)])
        style.configure("TCombobox", fieldbackground="#3a3a3a", background=accent_btn, foreground=fg_text)
        style.configure("TEntry", fieldbackground="#3a3a3a", foreground=fg_text, insertcolor=fg_text)
    else:
        root.configure(bg="#f0f0f0")
        style.theme_use('default')


# --- Modal Dialogs ---
class ConfigDialog(tk.Toplevel):
    """Modal preferences dialog including theme configuration."""
    def __init__(self, parent, current_cfg):
        super().__init__(parent)
        self.title("VirtualSTEMExplorerTK - Preferences")
        self.resizable(False, False)
        self.grab_set()

        self.cfg = current_cfg.copy()
        self.updated = False

        apply_theme_styles(self, self.cfg.get("theme", "dark"))

        f_main = ttk.Frame(self, padding=10)
        f_main.pack(fill=tk.BOTH, expand=True)

        # 1. UI Appearance / Theme
        f_theme = ttk.LabelFrame(f_main, text="GUI Appearance", padding=8)
        f_theme.grid(row=0, column=0, columnspan=2, sticky='ew', pady=5)

        ttk.Label(f_theme, text="Theme Scheme:").grid(row=0, column=0, sticky='w', padx=4)
        self.var_theme = tk.StringVar(value=self.cfg.get("theme", "dark"))
        cb_theme = ttk.Combobox(f_theme, textvariable=self.var_theme, values=["dark", "light"], width=10, state="readonly")
        cb_theme.grid(row=0, column=1, sticky='w', padx=4)

        # 2. Calibration DB Path
        f_db = ttk.LabelFrame(f_main, text="Microscope Calibration Database", padding=8)
        f_db.grid(row=1, column=0, columnspan=2, sticky='ew', pady=5)

        ttk.Label(f_db, text="JSON DB Path:").grid(row=0, column=0, sticky='w')
        self.var_db_path = tk.StringVar(value=self.cfg.get("cal_db_path", ""))
        ttk.Entry(f_db, textvariable=self.var_db_path, width=45).grid(row=1, column=0, padx=2)
        ttk.Button(f_db, text="Browse...", command=self.browse_db).grid(row=1, column=1, padx=2)

        # 3. Display Defaults
        f_disp = ttk.LabelFrame(f_main, text="Colormap & Mask Defaults", padding=8)
        f_disp.grid(row=2, column=0, columnspan=2, sticky='ew', pady=5)

        cmap_opts = ['gray', 'gray_r', 'inferno', 'magma', 'viridis', 'plasma', 'turbo', 'cividis', 'hsv', 'twilight', 'twilight_shifted']

        ttk.Label(f_disp, text="Left Cmap:").grid(row=0, column=0, sticky='e', padx=4)
        self.var_l_cmap = tk.StringVar(value=self.cfg.get("left_cmap", "gray"))
        ttk.Combobox(f_disp, textvariable=self.var_l_cmap, values=cmap_opts, width=10, state="readonly").grid(row=0, column=1, padx=4)

        ttk.Label(f_disp, text="Right Cmap:").grid(row=0, column=2, sticky='e', padx=4)
        self.var_r_cmap = tk.StringVar(value=self.cfg.get("right_cmap", "turbo"))
        ttk.Combobox(f_disp, textvariable=self.var_r_cmap, values=cmap_opts, width=10, state="readonly").grid(row=0, column=3, padx=4)

        ttk.Label(f_disp, text="Default dx/dy:").grid(row=1, column=0, sticky='e', padx=4, pady=4)
        self.var_dx = tk.StringVar(value=str(self.cfg.get("dx", 5)))
        self.var_dy = tk.StringVar(value=str(self.cfg.get("dy", 5)))
        ttk.Entry(f_disp, textvariable=self.var_dx, width=4).grid(row=1, column=1, sticky='w', padx=4)

        ttk.Label(f_disp, text="Default r_in/r_out:").grid(row=1, column=2, sticky='e', padx=4, pady=4)
        self.var_rin = tk.StringVar(value=str(self.cfg.get("r_in", 0.0)))
        self.var_rout = tk.StringVar(value=str(self.cfg.get("r_out", 25.0)))
        ttk.Entry(f_disp, textvariable=self.var_rin, width=4).grid(row=1, column=3, sticky='w', padx=4)

        # 4. Default Binning
        f_bin = ttk.LabelFrame(f_main, text="Default File Import Binning", padding=8)
        f_bin.grid(row=3, column=0, columnspan=2, sticky='ew', pady=5)

        ttk.Label(f_bin, text="bin_scan (Y, X):").grid(row=0, column=0, sticky='e', padx=4)
        self.var_bs_y = tk.StringVar(value=str(self.cfg.get("bin_scan_y", 2)))
        self.var_bs_x = tk.StringVar(value=str(self.cfg.get("bin_scan_x", 2)))
        ttk.Entry(f_bin, textvariable=self.var_bs_y, width=4).grid(row=0, column=1, padx=2)
        ttk.Entry(f_bin, textvariable=self.var_bs_x, width=4).grid(row=0, column=2, padx=2)

        ttk.Label(f_bin, text="bin_det (Y, X):").grid(row=1, column=0, sticky='e', padx=4, pady=4)
        self.var_bd_y = tk.StringVar(value=str(self.cfg.get("bin_det_y", 1)))
        self.var_bd_x = tk.StringVar(value=str(self.cfg.get("bin_det_x", 1)))
        ttk.Entry(f_bin, textvariable=self.var_bd_y, width=4).grid(row=1, column=1, padx=2)
        ttk.Entry(f_bin, textvariable=self.var_bd_x, width=4).grid(row=1, column=2, padx=2)

        # Action Buttons
        f_btn = ttk.Frame(f_main)
        f_btn.grid(row=4, column=0, columnspan=2, pady=8)
        ttk.Button(f_btn, text="Save & Apply", command=self.on_save).pack(side=tk.LEFT, padx=5)
        ttk.Button(f_btn, text="Cancel", command=self.destroy).pack(side=tk.LEFT, padx=5)

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
            self.cfg["bin_scan_y"] = int(self.var_bs_y.get())
            self.cfg["bin_scan_x"] = int(self.var_bs_x.get())
            self.cfg["bin_det_y"] = int(self.var_bd_y.get())
            self.cfg["bin_det_x"] = int(self.var_bd_x.get())

            save_config(self.cfg)
            self.updated = True
            self.destroy()
        except ValueError as e:
            messagebox.showerror("Validation Error", f"Please check numerical input fields:\n{e}", parent=self)


class FileImportDialog(tk.Toplevel):
    def __init__(self, parent, initial_cfg):
        super().__init__(parent)
        self.title("DECTRIS File Import Options")
        self.resizable(False, False)
        self.grab_set()

        apply_theme_styles(self, initial_cfg.get("theme", "dark"))
        self.result = None

        ttk.Label(self, text="Select Master HDF5 File:").grid(row=0, column=0, columnspan=2, sticky='w', padx=10, pady=5)
        self.var_filepath = tk.StringVar()
        ttk.Entry(self, textvariable=self.var_filepath, width=50).grid(row=1, column=0, padx=10, pady=2)
        ttk.Button(self, text="Browse...", command=self.browse_file).grid(row=1, column=1, padx=5, pady=2)

        f_bin = ttk.LabelFrame(self, text="Binning Settings", padding=10)
        f_bin.grid(row=2, column=0, columnspan=2, sticky='ew', padx=10, pady=10)

        ttk.Label(f_bin, text="Real Space Binning (bin_scan Y, X):").grid(row=0, column=0, sticky='e', padx=5)
        self.var_bs_y = tk.StringVar(value=str(initial_cfg.get("bin_scan_y", 2)))
        self.var_bs_x = tk.StringVar(value=str(initial_cfg.get("bin_scan_x", 2)))
        ttk.Entry(f_bin, textvariable=self.var_bs_y, width=5).grid(row=0, column=1, padx=2)
        ttk.Entry(f_bin, textvariable=self.var_bs_x, width=5).grid(row=0, column=2, padx=2)

        ttk.Label(f_bin, text="Reciprocal Space Binning (bin_det Y, X):").grid(row=1, column=0, sticky='e', padx=5, pady=5)
        self.var_bd_y = tk.StringVar(value=str(initial_cfg.get("bin_det_y", 1)))
        self.var_bd_x = tk.StringVar(value=str(initial_cfg.get("bin_det_x", 1)))
        ttk.Entry(f_bin, textvariable=self.var_bd_y, width=5).grid(row=1, column=1, padx=2)
        ttk.Entry(f_bin, textvariable=self.var_bd_x, width=5).grid(row=1, column=2, padx=2)

        f_btn = ttk.Frame(self)
        f_btn.grid(row=3, column=0, columnspan=2, pady=10)
        ttk.Button(f_btn, text="Load Data", command=self.on_ok).pack(side=tk.LEFT, padx=5)
        ttk.Button(f_btn, text="Cancel", command=self.destroy).pack(side=tk.LEFT, padx=5)

    def browse_file(self):
        filename = filedialog.askopenfilename(
            title="Select DECTRIS Master HDF5 File",
            filetypes=[("HDF5 Files", "*.h5 *.hdf5"), ("All Files", "*.*")]
        )
        if filename:
            self.var_filepath.set(filename)

    def on_ok(self):
        filepath = self.var_filepath.get().strip()
        if not filepath or not os.path.exists(filepath):
            messagebox.showerror("Invalid File", "Please select a valid DECTRIS master file.", parent=self)
            return

        try:
            bs_y = int(self.var_bs_y.get())
            bs_x = int(self.var_bs_x.get())
            bd_y = int(self.var_bd_y.get())
            bd_x = int(self.var_bd_x.get())
        except ValueError:
            messagebox.showerror("Invalid Binning", "Binning factors must be integer values.", parent=self)
            return

        filedir, filename = os.path.split(filepath)
        basename, suffix = os.path.splitext(filename)

        self.result = {
            "full_path": filepath,
            "path": filedir + "/",
            "filebasename": basename,
            "filesuffix": suffix,
            "bin_scan": (bs_y, bs_x),
            "bin_det": (bd_y, bd_x)
        }
        self.destroy()


# --- Main Application Explorer ---
def VirtualSTEMExplorerTK(myset=None, initial_filepath=None):
    config = load_config()
    target_dict = {}

    root = tk.Tk()
    root.title("VirtualSTEMExplorerTK")
    root.geometry("1400x950")

    # Apply Selected Theme
    apply_theme_styles(root, config.get("theme", "dark"))

    # Matplotlib Theme Switcher
    if config.get("theme", "dark") == "dark":
        plt.style.use('dark_background')
    else:
        plt.style.use('default')

        # Add an About handler:
    def show_about_dialog():
        messagebox.showinfo(
            "About VirtualSTEMExplorerTK",
            "VirtualSTEMExplorerTK v1.0\n"
            "Interactive 4D-STEM Analysis Suite\n\n"
            "Copyright © 2026 L. Houben - Weizmann Institute of Science.\n"
            ""
        )
        
    state = {
        'myset': myset,
        'path': "",
        'filebasename': "VirtualSTEMExplorer_Export",
        'filesuffix': ".h5",
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
    top_bar = ttk.Frame(root, padding="6")
    top_bar.pack(side=tk.TOP, fill=tk.X)

    btn_open = ttk.Button(top_bar, text="📁 Open DECTRIS File", command=lambda: open_file_dialog_action())
    btn_open.pack(side=tk.LEFT, padx=4)

    btn_config = ttk.Button(top_bar, text="⚙️ Settings...", command=lambda: open_config_dialog_action())
    btn_config.pack(side=tk.LEFT, padx=4)

    lbl_status = ttk.Label(top_bar, text="No dataset loaded. Click 'Open DECTRIS File' to start.", font=("Arial", 10, "italic"))
    lbl_status.pack(side=tk.LEFT, padx=12)

    canvas_frame = ttk.Frame(root)
    canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.subplots_adjust(top=0.88, bottom=0.15, left=0.08, right=0.92, wspace=0.35)

    canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
    canvas.draw()
    canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    toolbar = NavigationToolbar2Tk(canvas, canvas_frame)
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

        state['x'] = scanx // 2
        state['y'] = scany // 2
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
        btn_save.config(text=f"Save #{next_idx:03d}")
        sync_target_dict()
        canvas.draw()

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
            apply_theme_styles(root, config.get("theme", "dark"))
            if config.get("theme", "dark") == "dark":
                plt.style.use('dark_background')
            else:
                plt.style.use('default')

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

            config['bin_scan_y'], config['bin_scan_x'] = res['bin_scan']
            config['bin_det_y'], config['bin_det_x'] = res['bin_det']
            save_config(config)

            try:
                new_set = nbed.pyNBED()
                args = {
                    'scan_shape': (256, 256),
                    'bin_scan': res['bin_scan'],
                    'bin_det': res['bin_det']
                }
                new_set.LoadFile(res['full_path'], type='DECTRIS', **args)
                state['myset'] = new_set

                if os.path.exists(config["cal_db_path"]):
                    cal_mgr = nbed.MicroscopeCalibrationManager(config["cal_db_path"])
                    samp, qsamp = cal_mgr.get_calibration_from_metadata(new_set.metadata, type='DECTRIS')
                    state['re_samp'] = samp if samp else 1.0
                    state['rec_samp'] = qsamp if qsamp else 1.0

                rebuild_plot()
                lbl_status.config(text=f"Loaded: {res['filebasename']}{res['filesuffix']}")
            except Exception as e:
                messagebox.showerror("Loading Error", f"Failed to load file:\n{e}")

    # Menu Bar
    menu_bar = tk.Menu(root)
    file_menu = tk.Menu(menu_bar, tearoff=0)
    file_menu.add_command(label="Open DECTRIS Master File...", command=open_file_dialog_action)
    file_menu.add_separator()
    file_menu.add_command(label="Settings...", command=open_config_dialog_action)
    # Add to menu bar:
    file_menu.add_command(label="About VirtualSTEMExplorerTK", command=show_about_dialog)
    file_menu.add_separator()
    file_menu.add_command(label="Exit", command=root.quit)
    menu_bar.add_cascade(label="File", menu=file_menu)
    root.config(menu=menu_bar)

    # Control Dock Frame
    ctrl_frame = ttk.Frame(root, padding="5")
    ctrl_frame.pack(side=tk.BOTTOM, fill=tk.X)

    # 1. Left Contrast Controls
    f_left = ttk.LabelFrame(ctrl_frame, text="Left Panel Controls", padding="5")
    f_left.pack(side=tk.LEFT, fill=tk.Y, padx=4, pady=2)

    ttk.Label(f_left, text="Clip Low/High:").grid(row=0, column=0, sticky='w')
    var_l_clip_low = tk.StringVar(value="0.0")
    var_l_clip_high = tk.StringVar(value="100.0")
    ttk.Entry(f_left, textvariable=var_l_clip_low, width=5).grid(row=0, column=1, padx=1)
    ttk.Entry(f_left, textvariable=var_l_clip_high, width=5).grid(row=0, column=2, padx=1)

    ttk.Label(f_left, text="Gamma:").grid(row=1, column=0, sticky='w')
    var_l_gamma = tk.StringVar(value="1.0")
    ttk.Entry(f_left, textvariable=var_l_gamma, width=5).grid(row=1, column=1, padx=1)

    ttk.Button(f_left, text="Apply", command=update_left).grid(row=1, column=2, padx=2)

    ttk.Label(f_left, text="Cmap:").grid(row=2, column=0, sticky='w')
    var_l_cmap = tk.StringVar(value=state['left_cmap'])
    cmap_opts_l = ['gray', 'gray_r', 'inferno', 'magma', 'viridis', 'plasma', 'turbo', 'cividis', 'hsv', 'twilight', 'twilight_shifted']
    cb_l_cmap = ttk.Combobox(f_left, textvariable=var_l_cmap, values=cmap_opts_l, width=10, state="readonly")
    cb_l_cmap.grid(row=2, column=1, columnspan=2, pady=2)

    def on_l_cmap_select(evt):
        state['left_cmap'] = var_l_cmap.get()
        config['left_cmap'] = state['left_cmap']
        save_config(config)
        if state['myset']:
            artists['im1'].set_cmap(state['left_cmap'])
            sync_target_dict()
            update_left()

    cb_l_cmap.bind("<<ComboboxSelected>>", on_l_cmap_select)

    # 2. Mask Mode
    f_mode = ttk.LabelFrame(ctrl_frame, text="Mask Mode", padding="5")
    f_mode.pack(side=tk.LEFT, fill=tk.Y, padx=4, pady=2)

    var_mode = tk.StringVar(value=state['calc_mode'])
    modes = ['Sum', 'Variance', 'Fluctuation', 'COM Mag', 'COM Azimuth']

    def on_mode_change():
        state['calc_mode'] = var_mode.get()
        if state['calc_mode'] == 'COM Azimuth':
            state['left_cmap'] = 'hsv'
            var_l_cmap.set('hsv')
            if state['myset']: artists['im1'].set_cmap('hsv')
        sync_target_dict()
        refresh_detector_mask()

    for i, m in enumerate(modes):
        ttk.Radiobutton(f_mode, text=m, value=m, variable=var_mode, command=on_mode_change).grid(row=i % 3, column=i // 3, sticky='w')

    # 3. Right Contrast
    f_right = ttk.LabelFrame(ctrl_frame, text="Right Panel Controls", padding="5")
    f_right.pack(side=tk.LEFT, fill=tk.Y, padx=4, pady=2)

    ttk.Label(f_right, text="Clip Low/High:").grid(row=0, column=0, sticky='w')
    var_r_clip_low = tk.StringVar(value="0.0")
    var_r_clip_high = tk.StringVar(value="100.0")
    ttk.Entry(f_right, textvariable=var_r_clip_low, width=5).grid(row=0, column=1, padx=1)
    ttk.Entry(f_right, textvariable=var_r_clip_high, width=5).grid(row=0, column=2, padx=1)

    ttk.Label(f_right, text="Gamma:").grid(row=1, column=0, sticky='w')
    var_r_gamma = tk.StringVar(value="1.0")
    ttk.Entry(f_right, textvariable=var_r_gamma, width=5).grid(row=1, column=1, padx=1)

    ttk.Button(f_right, text="Apply", command=update_right).grid(row=1, column=2, padx=2)

    ttk.Label(f_right, text="Cmap:").grid(row=2, column=0, sticky='w')
    var_r_cmap = tk.StringVar(value=state['right_cmap'])
    cmap_opts_r = ['gray', 'gray_r', 'inferno', 'magma', 'viridis', 'plasma', 'turbo', 'cividis']
    cb_r_cmap = ttk.Combobox(f_right, textvariable=var_r_cmap, values=cmap_opts_r, width=10, state="readonly")
    cb_r_cmap.grid(row=2, column=1, columnspan=2, pady=2)

    def on_r_cmap_select(evt):
        state['right_cmap'] = var_r_cmap.get()
        config['right_cmap'] = state['right_cmap']
        save_config(config)
        if state['myset']:
            artists['im2'].set_cmap(state['right_cmap'])
            sync_target_dict()
            update_right()

    cb_r_cmap.bind("<<ComboboxSelected>>", on_r_cmap_select)

    # 4. Mask & ROI Size Settings
    f_params = ttk.LabelFrame(ctrl_frame, text="ROI & Mask Radii", padding="5")
    f_params.pack(side=tk.LEFT, fill=tk.Y, padx=4, pady=2)

    ttk.Label(f_params, text="dx:").grid(row=0, column=0, sticky='e')
    var_dx = tk.StringVar(value=str(state['dx']))
    ttk.Entry(f_params, textvariable=var_dx, width=4).grid(row=0, column=1)

    ttk.Label(f_params, text="dy:").grid(row=0, column=2, sticky='e')
    var_dy = tk.StringVar(value=str(state['dy']))
    ttk.Entry(f_params, textvariable=var_dy, width=4).grid(row=0, column=3)

    def set_roi_cmd():
        config['dx'] = int(var_dx.get())
        config['dy'] = int(var_dy.get())
        save_config(config)
        refresh_spatial_roi()

    ttk.Button(f_params, text="Set ROI", command=set_roi_cmd).grid(row=0, column=4, padx=2)

    ttk.Label(f_params, text="r_in:").grid(row=1, column=0, sticky='e')
    var_rin = tk.StringVar(value=str(state['r_in']))
    ttk.Entry(f_params, textvariable=var_rin, width=4).grid(row=1, column=1)

    ttk.Label(f_params, text="r_out:").grid(row=1, column=2, sticky='e')
    var_rout = tk.StringVar(value=str(state['r_out']))
    ttk.Entry(f_params, textvariable=var_rout, width=4).grid(row=1, column=3)

    def set_mask_cmd():
        config['r_in'] = float(var_rin.get())
        config['r_out'] = float(var_rout.get())
        save_config(config)
        refresh_detector_mask()

    ttk.Button(f_params, text="Set Mask", command=set_mask_cmd).grid(row=1, column=4, padx=2)

    # 5. Measurement Tools & Publication Export
    f_actions = ttk.LabelFrame(ctrl_frame, text="Tools & Export", padding="5")
    f_actions.pack(side=tk.LEFT, fill=tk.Y, padx=4, pady=2)

    btn_r1 = ttk.Button(f_actions, text="Ruler Real-Space")
    btn_r2 = ttk.Button(f_actions, text="Ruler Diffraction")

    def toggle_ruler1():
        if state['myset'] is None: return
        ruler1_state['active'] = not ruler1_state['active']
        ruler1_state['p1'] = None; ruler1_state['p2'] = None; ruler1_state['length'] = 0.0
        sync_target_dict()
        if ruler1_state['active']:
            btn_r1.config(text="Real ON")
            artists['ruler1_text'].set_visible(True)
            artists['ruler1_text'].set_text("Click Point A on Virtual Image...")
        else:
            btn_r1.config(text="Ruler Real-Space")
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
            btn_r2.config(text="Recip ON")
            artists['ruler2_text'].set_visible(True)
            artists['ruler2_text'].set_text("Click Point A on Diffraction Pattern...")
        else:
            btn_r2.config(text="Ruler Diffraction")
            artists['ruler2_line'].set_data([], [])
            artists['ruler2_text'].set_visible(False)
        canvas.draw_idle()

    btn_r1.config(command=toggle_ruler1)
    btn_r2.config(command=toggle_ruler2)
    btn_r1.grid(row=0, column=0, padx=2, pady=1)
    btn_r2.grid(row=0, column=1, padx=2, pady=1)

    def save_figure_callback():
        if state['myset'] is None:
            messagebox.showwarning("No Data", "Please load a dataset before saving.")
            return
        next_idx, filename = get_next_filename()
        try:
            fig.savefig(filename, bbox_inches='tight', pad_inches=0.05)
            next_idx_new, _ = get_next_filename()
            btn_save.config(text=f"Export Figure #{next_idx_new:03d}")
            messagebox.showinfo("Export Success", f"Successfully exported:\n{filename}")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to export figure:\n{e}")

    btn_save = ttk.Button(f_actions, text="Export Figure #001", command=save_figure_callback)
    btn_save.grid(row=1, column=0, columnspan=2, padx=2, pady=1)

    if initial_filepath and os.path.exists(initial_filepath):
        filedir, filename = os.path.split(initial_filepath)
        basename, suffix = os.path.splitext(filename)
        state['path'] = filedir + "/"
        state['filebasename'] = basename
        state['filesuffix'] = suffix

        myset_obj = nbed.pyNBED()
        args = {
            'scan_shape': (256, 256),
            'bin_scan': (config["bin_scan_y"], config["bin_scan_x"]),
            'bin_det': (config["bin_det_y"], config["bin_det_x"])
        }
        myset_obj.LoadFile(initial_filepath, type='DECTRIS', **args)
        state['myset'] = myset_obj

        if os.path.exists(config["cal_db_path"]):
            cal_mgr = nbed.MicroscopeCalibrationManager(config["cal_db_path"])
            samp, qsamp = cal_mgr.get_calibration_from_metadata(myset_obj.metadata, type='DECTRIS')
            state['re_samp'] = samp if samp else 1.0
            state['rec_samp'] = qsamp if qsamp else 1.0

        rebuild_plot()
        lbl_status.config(text=f"Loaded: {basename}{suffix}")

    root.mainloop()
    return target_dict


def main():

    VirtualSTEMExplorerTK(initial_filepath=None)

if __name__ == "__main__":
    main()

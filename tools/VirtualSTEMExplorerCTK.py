import os
import sys
import glob
import re
import json
import time
import pickle
import inspect
import threading
import concurrent.futures
from pathlib import Path
import numpy as np
from scipy.ndimage import median_filter

import tkinter as tk
from tkinter import filedialog, messagebox
import customtkinter as ctk

import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle, Circle, Polygon
from mpl_toolkits.axes_grid1 import make_axes_locatable

import nbed
import trackpy as tp
import ase.io
import gzip
import zipfile

try:
    import py4DSTEM
    from py4DSTEM.process.diffraction import Crystal
    HAS_PY4D = True
except ImportError:
    HAS_PY4D = False

try:
    import tifffile
    HAS_TIFFFILE = True
except ImportError:
    HAS_TIFFFILE = False


# ==============================================================================
# Path & Configuration Management
# ==============================================================================
def get_app_config_path() -> Path:
    if sys.platform == "win32":
        base_dir = Path(os.getenv("APPDATA", Path.home() / "AppData" / "Roaming"))
    else:
        base_dir = Path(os.getenv("XDG_CONFIG_HOME", Path.home() / ".config"))
    
    config_dir = base_dir / "VirtualSTEMExplorerTK"
    config_dir.mkdir(parents=True, exist_ok=True)
    return config_dir / "VirtualSTEMExplorerCTK_config.json"

CONFIG_FILE = get_app_config_path()

DEFAULT_RING_PRESETS = {
    "Ice (P63mm)": [2.571, 2.73, 3.75, 4.45, 5.13, 5.22],
    "Gold (Au)": [2.12, 2.45, 3.47, 4.06, 4.24, 4.90],
    "Silicon (Si)": [1.57, 2.56, 3.01, 3.62, 3.95, 4.44]
}

DEFAULT_PEAK_PARAMS = {
    'fit_scale_linear': False,
    'feature_size': 7,
    'feature_minmass': 2.0,
    'feature_separation': 8,
    'feature_percentile': 0.0,
    'noise_size': 1.0,
    'feature_threshold': 0.5,
    'smooth_size': 0,
    'frame_cmin': 1.0,
    'frame_cmax': 10.0,
    'conv2D': False,
    'frame_cutoff': 0.0,
    'peak_keep_percent': 100.0,
    'edge_margin': 5,
    'exclude_center_radius': 0.5
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
    "eps22": 1.0,
    "detector_preset": "ELA",
    "fit_scale_linear": False,
    "sigma_excitation_error": 0.02,
    "larmor_offset_deg": 0.0  # <-- Added
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


# Try importing colorcet for Peter Kovesi's perceptually uniform cyclic colormaps
try:
    import colorcet as cc
    HAS_COLORCET = True
except ImportError:
    HAS_COLORCET = False

def get_available_cyclic_colormaps():
    """
    Returns an ordered dictionary of colormap labels to Matplotlib/Colorcet identifiers,
    including perceptual cyclic palettes, standard cyclic palettes, and standard Matplotlib maps.
    """
    cmaps = {
        # Perceptually uniform cyclic palettes
        "Twilight (Perceptual)": "twilight",
        "Twilight Shifted (Perceptual)": "twilight_shifted",
    }
    
    if HAS_COLORCET:
        cmaps.update({
            "CET Colorwheel (Perceptual)": "cet_colorwheel",
            "CET Isoluminant (Perceptual)": "cet_isolum",
            "CET Rainbow (Perceptual)": "cet_cyclic_mygbm_30_95_c78",
            "CET MRYBM (Perceptual)": "cet_cyclic_mrybm_35_75_c68"
        })

    # Standard Matplotlib cyclic and sequential colormaps
    cmaps.update({
        "HSV (Standard Cyclic)": "hsv",
        "Gist Rainbow": "gist_rainbow",
        "Turbo": "turbo",
        "Viridis": "viridis",
        "Plasma": "plasma",
        "Inferno": "inferno",
        "Magma": "magma",
        "Cividis": "cividis",
        "Coolwarm": "coolwarm",
        "Ocean": "ocean",
        "Spectral": "Spectral",
        "Rainbow": "rainbow"
    })
    return cmaps

def get_available_display_colormaps():
    """
    Returns a list of high-quality sequential, diverging, and perceptually uniform
    colormaps for real-space and reciprocal-space 2D intensity visualizations.
    """
    base_cmaps = [
        # Perceptually uniform sequential (Matplotlib defaults)
        'viridis', 'plasma', 'inferno', 'magma', 'cividis',
        # Standard analytical grayscale
        'gray', 'gray_r',
        # Cyclic / Angular / Phase
        'twilight', 'twilight_shifted', 'hsv',
        # Qualitative / Rainbow / High-Dynamic Range
        'turbo', 'coolwarm', 'RdBu_r'
    ]
    
    # If Peter Kovesi's colorcet package is installed, add perceptually linear colormaps
    if HAS_COLORCET:
        base_cmaps.extend([
            'cet_fire',       # Perceptually uniform black-red-yellow-white
            'cet_gray',       # Strict linear luminance grayscale
            'cet_bmy',        # Blue-Magenta-Yellow perceptual
            'cet_rainbow4',   # Perceptually balanced rainbow
            'cet_isolum'      # Isoluminant
        ])
        
    return base_cmaps

# ==============================================================================
# Peak Detection Algorithm
# ==============================================================================
def PeakDetection(img, params=None, origin='lower', lin=False):        
    if params is None:
        params = DEFAULT_PEAK_PARAMS.copy()

    feat_sz = int(params.get('feature_size', 7))
    if feat_sz % 2 == 0:
        feat_sz += 1

    feat_minmass = float(params.get('feature_minmass', 2.0))
    feat_sep = int(params.get('feature_separation', feat_sz + 1))
    feat_perc = float(params.get('feature_percentile', 0.0))
    noise_sz = float(params.get('noise_size', 1.0))
    thresh = float(params.get('feature_threshold', 0.5))
    smooth_sz = params.get('smooth_size', None)
    if smooth_sz is not None and smooth_sz <= 0:
        smooth_sz = None

    conv2D = bool(params.get('conv2D', False))

    proc_img = np.array(img, dtype=np.float64, copy=True)
    cutoff = float(params.get("frame_cutoff", 0.0))
    if cutoff > 0.0:
        proc_img[proc_img < cutoff] = 0.0
    else:
        proc_img[proc_img < 0.0] = 0.0

    framepeaklist = list()
    halfframedim = [proc_img.shape[1] / 2.0, proc_img.shape[0] / 2.0]

    if conv2D and hasattr(nbed, 'convolve2D'):
        conv_kernel = 1.0 / 9.0 * np.ones((3, 3))
        filtim = nbed.convolve2D(proc_img, conv_kernel, padding=2)
        filtim[filtim < 0.0] = 0.0
    else:
        filtim = proc_img

    target_img = filtim if lin else np.log1p(filtim)

    f = tp.locate(
        target_img,
        diameter=feat_sz,
        noise_size=noise_sz,
        threshold=thresh,
        minmass=feat_minmass,
        percentile=feat_perc,
        separation=feat_sep,
        smoothing_size=smooth_sz,
        invert=False,
        engine="python"
    )

    if f is not None and len(f) > 0:
        q = np.hypot(f['x'].values - halfframedim[0], f['y'].values - halfframedim[1])
        f.insert(1, "q", q, True)
        f = f.sort_values(by=['q'], ascending=True)

        framepeaklist.append({
            'x': f['x'].values - halfframedim[0],
            'y': f['y'].values - halfframedim[1],
            'q': f['q'].values,
            'mass': f['mass'].values,
            'size': f['size'].values,
            'raw_mass': f['raw_mass'].values
        })

    return framepeaklist


# ==============================================================================
# Multiprocessing Standalone Workers (macOS & Windows Spawn Safe)
# ==============================================================================
def _strain_worker_chunk(task_data):
    """
    Worker task for batch strain mapping.
    Processes chunked 2D diffraction frames through PeakDetection and least-squares deformation fit.
    """
    results = []
    center_search_rad = 25.0

    for item in task_data:
        (py, px, frame, pars, use_lin, geom_cx, geom_cy,
         init_cqx, init_cqy, rec_samp, origin_upper, sim_coords, r_cutoff) = item

        peaks = PeakDetection(frame, params=pars, origin='upper' if origin_upper else 'lower', lin=use_lin)
        if not peaks:
            results.append((py, px, np.nan, np.nan, np.nan, np.nan))
            continue

        res_dict = peaks[0] if isinstance(peaks, list) else peaks
        px_rel_x = np.atleast_1d(res_dict.get('x', []))
        px_rel_y = np.atleast_1d(res_dict.get('y', []))
        masses = np.atleast_1d(res_dict.get('mass', np.ones_like(px_rel_x)))
        raw_masses = np.atleast_1d(res_dict.get('raw_mass', masses))

        feat_sz = int(pars.get('feature_size', 7))
        if feat_sz % 2 == 0:
            feat_sz += 1
        spot_area = max(1.0, (np.pi / 4.0) * (feat_sz ** 2))

        if not use_lin:
            mean_log = np.maximum(0.0, raw_masses / spot_area)
            lin_intensities = np.expm1(mean_log) * spot_area
        else:
            lin_intensities = raw_masses.copy()

        det_px_x = px_rel_x + geom_cx
        det_px_y = px_rel_y + geom_cy

        dists_to_center = np.hypot(det_px_x - init_cqx, det_px_y - init_cqy)
        near_center = dists_to_center < center_search_rad

        if np.any(near_center):
            central_idx = np.where(near_center)[0][np.argmax(lin_intensities[near_center])]
            frame_cqx = float(det_px_x[central_idx])
            frame_cqy = float(det_px_y[central_idx])
        else:
            frame_cqx, frame_cqy = init_cqx, init_cqy

        edge_margin = max(0, int(pars.get('edge_margin', 5)))
        det_dim_y, det_dim_x = frame.shape

        valid_edge = (
            (det_px_x >= edge_margin) & (det_px_x < det_dim_x - edge_margin) &
            (det_px_y >= edge_margin) & (det_px_y < det_dim_y - edge_margin)
        )

        det_px_x = det_px_x[valid_edge]
        det_px_y = det_px_y[valid_edge]

        qx_exp = (det_px_x - frame_cqx) * rec_samp
        qy_exp = (frame_cqy - det_px_y) * rec_samp if origin_upper else (det_px_y - frame_cqy) * rec_samp

        exp_r = np.hypot(qx_exp, qy_exp)
        diff_mask = exp_r > 0.5

        if np.sum(diff_mask) < 3:
            results.append((py, px, np.nan, np.nan, np.nan, np.nan))
            continue

        exp_coords = np.column_stack([qx_exp[diff_mask], qy_exp[diff_mask]])
        matched_exp, matched_sim = [], []

        for pt in exp_coords:
            dists = np.linalg.norm(sim_coords - pt, axis=1)
            min_i = int(np.argmin(dists))
            if dists[min_i] <= r_cutoff:
                matched_exp.append(pt)
                matched_sim.append(sim_coords[min_i])

        if len(matched_exp) >= 3:
            X, Y = np.array(matched_sim), np.array(matched_exp)
            try:
                A, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
                T = np.linalg.inv(A.T).T
                results.append((
                    py, px,
                    float(T[0, 0] - 1.0),
                    float(T[1, 1] - 1.0),
                    float(0.5 * (T[0, 1] + T[1, 0])),
                    float(0.5 * (T[0, 1] - T[1, 0]))
                ))
            except Exception:
                results.append((py, px, np.nan, np.nan, np.nan, np.nan))
        else:
            results.append((py, px, np.nan, np.nan, np.nan, np.nan))

    return results


def _orient_worker_chunk(task_data):
    """
    Worker task for batch orientation indexing.
    Processes chunked 2D diffraction frames through PeakDetection and ACOM template matching.
    """
    results = []
    center_search_rad = 25.0

    for item in task_data:
        (py, px, frame, pars, use_lin, geom_cx, geom_cy,
         init_cqx, init_cqy, rec_samp, origin_upper, M_active, acom_mgr) = item

        peaks = PeakDetection(frame, params=pars, origin='upper' if origin_upper else 'lower', lin=use_lin)
        if not peaks:
            results.append((py, px, 0.0, np.nan, np.nan, None))
            continue

        res_dict = peaks[0] if isinstance(peaks, list) else peaks
        px_rel_x = np.atleast_1d(res_dict.get('x', []))
        px_rel_y = np.atleast_1d(res_dict.get('y', []))
        masses = np.atleast_1d(res_dict.get('mass', np.ones_like(px_rel_x)))
        raw_masses = np.atleast_1d(res_dict.get('raw_mass', masses))

        feat_sz = int(pars.get('feature_size', 7))
        if feat_sz % 2 == 0:
            feat_sz += 1
        spot_area = max(1.0, (np.pi / 4.0) * (feat_sz ** 2))

        if not use_lin:
            mean_log = np.maximum(0.0, raw_masses / spot_area)
            lin_intensities = np.expm1(mean_log) * spot_area
        else:
            lin_intensities = raw_masses.copy()

        det_px_x = px_rel_x + geom_cx
        det_px_y = px_rel_y + geom_cy

        dists_to_center = np.hypot(det_px_x - init_cqx, det_px_y - init_cqy)
        near_center = dists_to_center < center_search_rad
        if np.any(near_center):
            central_idx = np.where(near_center)[0][np.argmax(lin_intensities[near_center])]
            frame_cqx = float(det_px_x[central_idx])
            frame_cqy = float(det_px_y[central_idx])
        else:
            frame_cqx, frame_cqy = init_cqx, init_cqy

        edge_margin = max(0, int(pars.get('edge_margin', 5)))
        r_center_excl = float(pars.get('exclude_center_radius', 0.5))
        det_dim_y, det_dim_x = frame.shape

        frame_peaks = []
        for xp, yp, m, lm in zip(det_px_x, det_px_y, masses, lin_intensities):
            if xp < edge_margin or xp >= det_dim_x - edge_margin or yp < edge_margin or yp >= det_dim_y - edge_margin:
                continue
            qx = (xp - frame_cqx) * rec_samp
            qy = (frame_cqy - yp) * rec_samp if origin_upper else (yp - frame_cqy) * rec_samp
            if np.hypot(qx, qy) < r_center_excl:
                continue
            frame_peaks.append({'qx': qx, 'qy': qy, 'intensity': lm, 'mass': m})

        keep_pct = float(pars.get('peak_keep_percent', 100.0))
        if keep_pct < 100.0 and len(frame_peaks) > 3:
            frame_peaks.sort(key=lambda p: p['intensity'], reverse=True)
            n_keep = max(3, int(np.ceil(len(frame_peaks) * (keep_pct / 100.0))))
            frame_peaks = frame_peaks[:n_keep]

        if len(frame_peaks) >= 3:
            try:
                cand_list = acom_mgr.match_experimental_peaks(frame_peaks, num_matches=1, distortion_matrix=M_active)
                if cand_list and len(cand_list) > 0:
                    top = cand_list[0]
                    corr_val = float(top['correlation'])
                    angle_val = float(top['in_plane_angle'])
                    za = top['zone_axis']
                    za_id = float(abs(za[0]) * 100 + abs(za[1]) * 10 + abs(za[2]))

                    if top.get('matrix') is not None:
                        R_mat = top['matrix']
                    else:
                        theta_rad = np.radians(angle_val)
                        cos_t, sin_t = np.cos(theta_rad), np.sin(theta_rad)
                        R_z = np.array([
                            [cos_t, -sin_t, 0.0],
                            [sin_t,  cos_t, 0.0],
                            [0.0,    0.0,   1.0]
                        ])
                        za_v = np.asarray(za, dtype=np.float64)
                        z_norm = za_v / (np.linalg.norm(za_v) + 1e-9)
                        t_vec = np.array([1.0, 0.0, 0.0]) if abs(z_norm[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
                        x_norm = np.cross(t_vec, z_norm)
                        x_norm = x_norm / np.linalg.norm(x_norm)
                        y_norm = np.cross(z_norm, x_norm)
                        R_mat = np.column_stack([x_norm, y_norm, z_norm]) @ R_z

                    results.append((py, px, corr_val, angle_val, za_id, R_mat))
                    continue
            except Exception:
                pass

        results.append((py, px, 0.0, np.nan, np.nan, None))

    return results

def _batch_peak_worker_chunk(task_data):
    """
    Worker task for batch peak detection across scan locations.
    Linearizes intensities and extracts frame-level peak candidates.
    task_data: list of tuples (i, j, frame_2d, pars, origin, use_lin)
    """
    results = []
    for item in task_data:
        i, j, frame, pars, origin, use_lin = item
        peaks = PeakDetection(frame, params=pars, origin=origin, lin=use_lin)
        
        if peaks and len(peaks) > 0:
            res_dict = peaks[0]
            x_arr = np.asarray(res_dict.get('x', []), dtype=np.float64)
            y_arr = np.asarray(res_dict.get('y', []), dtype=np.float64)
            q_arr = np.asarray(res_dict.get('q', []), dtype=np.float64)
            masses = np.asarray(res_dict.get('mass', np.ones_like(x_arr)), dtype=np.float64)
            raw_masses = np.asarray(res_dict.get('raw_mass', masses), dtype=np.float64)
            sizes = np.asarray(res_dict.get('size', np.ones_like(x_arr)), dtype=np.float64)

            # Linearize intensities if detection was performed on log-scaled frames
            feat_sz = int(pars.get('feature_size', 7))
            if feat_sz % 2 == 0:
                feat_sz += 1
            spot_area = max(1.0, (np.pi / 4.0) * (feat_sz ** 2))

            if not use_lin:
                mean_log = np.maximum(0.0, raw_masses / spot_area)
                lin_intensities = np.expm1(mean_log) * spot_area
            else:
                lin_intensities = raw_masses.copy()

            results.append({
                'i': int(i),
                'j': int(j),
                'x': x_arr,
                'y': y_arr,
                'q': q_arr,
                'intensity': lin_intensities,
                'mass': masses,
                'size': sizes,
                'raw_mass': raw_masses
            })
        else:
            empty = np.array([], dtype=np.float64)
            results.append({
                'i': int(i),
                'j': int(j),
                'x': empty,
                'y': empty,
                'q': empty,
                'intensity': empty,
                'mass': empty,
                'size': empty,
                'raw_mass': empty
            })
    return results

# ==============================================================================
# ACOM & Kinematical Library Engine
# ==============================================================================
class ACOMManager:
    def __init__(self):
        self.crystal = None
        self.cif_path = ""
        self.library_path = ""
        self.plan_ready = False
        self.sigma_excitation_error = 0.02

    def build_library_from_cif(self, cif_path: str, accel_voltage: float = 200e3,
                               k_max: float = 2.0, angular_step_deg: float = 1.0,
                               sigma_excitation_error: float = 0.02) -> str:
        if not HAS_PY4D:
            raise RuntimeError("py4DSTEM is not installed in the active environment.")

        self.cif_path = cif_path
        self.sigma_excitation_error = sigma_excitation_error

        atoms = ase.io.read(cif_path)
        self.crystal = Crystal(
            numbers=atoms.numbers,
            cell=atoms.cell,
            positions=atoms.positions
        )
        self.crystal.accelerating_voltage = accel_voltage

        if hasattr(self.crystal, 'calculate_structure_factors'):
            sig_sf = inspect.signature(self.crystal.calculate_structure_factors)
            sf_kwargs = {}
            if 'k_max' in sig_sf.parameters:
                sf_kwargs['k_max'] = k_max
            elif 'kmax' in sig_sf.parameters:
                sf_kwargs['kmax'] = k_max
            self.crystal.calculate_structure_factors(**sf_kwargs)

        zone_axis_range = None
        if hasattr(self.crystal, 'get_zone_axis_range'):
            try:
                zone_axis_range = self.crystal.get_zone_axis_range()
            except Exception:
                zone_axis_range = None

        if zone_axis_range is None:
            try:
                import py4DSTEM.process.diffraction as p4d_diff
                if hasattr(p4d_diff, 'get_zone_axis_range'):
                    zone_axis_range = p4d_diff.get_zone_axis_range(self.crystal)
            except Exception:
                zone_axis_range = None

        if zone_axis_range is None or not hasattr(zone_axis_range, 'shape'):
            zone_axis_range = np.eye(3, dtype=np.float64)
        else:
            zone_axis_range = np.asarray(zone_axis_range, dtype=np.float64)

        sig_op = inspect.signature(self.crystal.orientation_plan)
        op_params = sig_op.parameters.keys()
        plan_kwargs = {}

        if 'zone_axis_range' in op_params:
            plan_kwargs['zone_axis_range'] = zone_axis_range

        if 'angle_step_zone_axis' in op_params:
            plan_kwargs['angle_step_zone_axis'] = angular_step_deg
        elif 'step_size' in op_params:
            plan_kwargs['step_size'] = angular_step_deg
        elif 'resolution' in op_params:
            plan_kwargs['resolution'] = angular_step_deg

        if 'accel_voltage' in op_params:
            plan_kwargs['accel_voltage'] = accel_voltage
        elif 'accelerating_voltage' in op_params:
            plan_kwargs['accelerating_voltage'] = accel_voltage

        if 'sigma_excitation_error' in op_params:
            plan_kwargs['sigma_excitation_error'] = sigma_excitation_error
        elif 'excitation_error' in op_params:
            plan_kwargs['excitation_error'] = sigma_excitation_error

        if 'progress_bar' in op_params:
            plan_kwargs['progress_bar'] = True

        self.crystal.orientation_plan(**plan_kwargs)

        num_templates = 0
        if hasattr(self.crystal, 'orientation_zone_axes'):
            za = getattr(self.crystal, 'orientation_zone_axes')
            if hasattr(za, 'shape') and len(za.shape) > 0:
                num_templates = za.shape[0]
            elif isinstance(za, (list, tuple)):
                num_templates = len(za)
        elif hasattr(self.crystal, 'orientations'):
            num_templates = len(self.crystal.orientations)

        self.plan_ready = True
        return f"Library built for {os.path.basename(cif_path)}: {num_templates} orientations generated (step={angular_step_deg}°, σ_s={sigma_excitation_error} 1/Å)."

    def save_library(self, filepath: str, compress: bool = True):
        if not self.plan_ready or self.crystal is None:
            raise RuntimeError("No orientation plan available to save.")

        save_dict = {
            'cif_path': self.cif_path,
            'crystal': self.crystal,
            'sigma_excitation_error': self.sigma_excitation_error
        }

        fn_lower = filepath.lower()

        if fn_lower.endswith('.zip') and not fn_lower.endswith('.gz'):
            pickled_bytes = pickle.dumps(save_dict, protocol=pickle.HIGHEST_PROTOCOL)
            with zipfile.ZipFile(filepath, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
                member_name = os.path.splitext(os.path.basename(filepath))[0] + ".pkl"
                zf.writestr(member_name, pickled_bytes)
        elif fn_lower.endswith(('.gz', '.pklz')):
            with gzip.open(filepath, 'wb', compresslevel=6) as f:
                pickle.dump(save_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        else:
            with open(filepath, 'wb') as f:
                pickle.dump(save_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

        self.library_path = filepath

    def load_library(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"File not found: {filepath}")

        fn_lower = filepath.lower()
        save_dict = None

        if fn_lower.endswith('.zip') or zipfile.is_zipfile(filepath):
            try:
                with zipfile.ZipFile(filepath, 'r') as zf:
                    pkl_names = [name for name in zf.namelist() if name.lower().endswith('.pkl')]
                    target_name = pkl_names[0] if pkl_names else zf.namelist()[0]
                    with zf.open(target_name, 'r') as member_file:
                        save_dict = pickle.load(member_file)
            except Exception:
                pass

        if save_dict is None:
            try:
                with gzip.open(filepath, 'rb') as f:
                    save_dict = pickle.load(f)
            except Exception:
                pass

        if save_dict is None:
            with open(filepath, 'rb') as f:
                save_dict = pickle.load(f)

        if not isinstance(save_dict, dict) or 'crystal' not in save_dict:
            raise ValueError("Corrupted orientation library file: missing 'crystal' entry.")

        self.crystal = save_dict['crystal']
        self.cif_path = save_dict.get('cif_path', '')
        self.sigma_excitation_error = save_dict.get('sigma_excitation_error', 0.02)
        self.plan_ready = True
        self.library_path = filepath

    def extract_lattice_zone_axis(self, item) -> np.ndarray:
        R = getattr(item, 'matrix', None)
        if R is None and isinstance(item, dict):
            R = item.get('matrix', item.get('orientation_matrix', None))

        if R is None:
            return np.array([0, 0, 1])

        R = np.asarray(R, dtype=np.float64)
        if R.ndim == 3:
            R = R[0]

        z_cart = R[:, 2]

        cell = getattr(self.crystal, 'cell', getattr(self.crystal, 'unitcell', None))
        if cell is not None and np.asarray(cell).shape == (3, 3):
            A = np.asarray(cell, dtype=np.float64)
            try:
                u_frac = np.linalg.inv(A.T) @ z_cart
            except Exception:
                u_frac = z_cart
        else:
            u_frac = z_cart

        nz = np.abs(u_frac)[np.abs(u_frac) > 1e-3]
        if len(nz) > 0:
            scaled = u_frac / np.min(nz)
            for mult in [1, 2, 3, 4, 5, 6]:
                test_int = scaled * mult
                if np.all(np.abs(test_int - np.round(test_int)) < 0.05):
                    uvw = np.round(test_int).astype(int)
                    break
            else:
                uvw = np.round(scaled).astype(int)
        else:
            uvw = np.array([0, 0, 1])

        nz_indices = np.nonzero(uvw)[0]
        if len(nz_indices) > 0 and uvw[nz_indices[0]] < 0:
            uvw = -uvw

        return uvw

    def match_experimental_peaks(self, peaks_list, num_matches: int = 3, distortion_matrix: np.ndarray = None):
        if not self.plan_ready or self.crystal is None:
            raise RuntimeError("Orientation library not loaded or generated.")
        if len(peaks_list) < 3:
            raise ValueError(f"At least 3 Bragg peaks required for indexing (found {len(peaks_list)}).")

        if distortion_matrix is None:
            M = np.eye(2, dtype=np.float64)
        else:
            M = np.asarray(distortion_matrix, dtype=np.float64)

        raw_qx_cart = np.array([p['qx'] for p in peaks_list], dtype=np.float64)
        raw_qy_cart = np.array([p['qy'] for p in peaks_list], dtype=np.float64)

        raw_cart = np.column_stack([raw_qx_cart, raw_qy_cart])
        corr_cart = (M @ raw_cart.T).T

        corr_qx_cart_A = (corr_cart[:, 0] * 0.1).astype(np.float32)
        corr_qy_cart_A = (corr_cart[:, 1] * 0.1).astype(np.float32)

        raw_intensity = np.array([
            p.get('intensity', p.get('raw_mass', p.get('mass', 1.0)))
            for p in peaks_list
        ], dtype=np.float32)

        max_int = np.max(raw_intensity) if np.max(raw_intensity) > 0 else 1.0
        norm_intensity = raw_intensity / max_int

        r_A = np.hypot(corr_qx_cart_A, corr_qy_cart_A)
        non_center_mask = r_A > 0.05

        if np.sum(non_center_mask) >= 3:
            use_qx_cart = corr_qx_cart_A[non_center_mask]
            use_qy_cart = corr_qy_cart_A[non_center_mask]
            int_arr = norm_intensity[non_center_mask]
        else:
            use_qx_cart, use_qy_cart, int_arr = corr_qx_cart_A, corr_qy_cart_A, norm_intensity

        n_pts = len(use_qx_cart)
        dtype = [('qx', np.float32), ('qy', np.float32), ('intensity', np.float32)]
        s_arr = np.zeros(n_pts, dtype=dtype)
        s_arr['qx'] = use_qy_cart
        s_arr['qy'] = use_qx_cart
        s_arr['intensity'] = int_arr

        pointlist = py4DSTEM.PointList(data=s_arr)
        try:
            pointlist.coordinates = ['qx', 'qy']
        except Exception:
            pass

        results = None
        try:
            results = self.crystal.match_single_pattern(pointlist, num_matches_return=int(num_matches))
        except Exception as e:
            print(f"[ACOM Diagnostic] match_single_pattern failed: {e}")
            return []

        if results is None:
            return []

        if isinstance(results, (list, tuple)):
            match_items = list(results)
        elif hasattr(results, 'matches'):
            match_items = list(results.matches)
        else:
            match_items = [results]

        n_candidates = min(len(match_items), int(num_matches))
        candidates = []
        for idx in range(n_candidates):
            item = match_items[idx]

            raw_corr = getattr(item, 'corr', getattr(item, 'correlation', 0.0))
            if hasattr(raw_corr, '__len__'):
                corr_val = float(np.ravel(raw_corr)[idx if len(raw_corr) > idx else 0])
            else:
                corr_val = float(raw_corr)

            R_mat = getattr(item, 'matrix', None)
            if R_mat is not None:
                R_mat = np.asarray(R_mat, dtype=np.float64)
                if R_mat.ndim == 3:
                    R_mat = R_mat[idx if len(R_mat) > idx else 0]

            uvw = self.extract_lattice_zone_axis(item)

            in_plane_rot = 0.0
            if R_mat is not None:
                in_plane_rot = float(np.degrees(np.arctan2(R_mat[1, 0], R_mat[0, 0])))

            sim_pattern = None
            try:
                if R_mat is not None:
                    sim_pattern = self.crystal.generate_diffraction_pattern(
                        orientation_matrix=R_mat,
                        sigma_excitation_error=self.sigma_excitation_error
                    )
                else:
                    sim_pattern = self.crystal.generate_diffraction_pattern(
                        zone_axis_lattice=uvw,
                        sigma_excitation_error=self.sigma_excitation_error
                    )
            except Exception:
                try:
                    sim_pattern = self.crystal.generate_diffraction_pattern(zone_axis_lattice=uvw)
                except Exception:
                    pass

            if sim_pattern is not None:
                sim_qx_cart, sim_qy_cart, sim_hkl, sim_I = None, None, [], None
                if hasattr(sim_pattern, 'data') and hasattr(sim_pattern.data, 'dtype') and sim_pattern.data.dtype.names is not None:
                    fnames = sim_pattern.data.dtype.names

                    if 'qx' in fnames and 'qy' in fnames:
                        sim_qx_cart = np.asarray(sim_pattern.data['qy'], dtype=np.float64) * 10.0
                        sim_qy_cart = np.asarray(sim_pattern.data['qx'], dtype=np.float64) * 10.0

                    if 'h' in fnames and 'k' in fnames and 'l' in fnames:
                        h = sim_pattern.data['h'].astype(int)
                        k = sim_pattern.data['k'].astype(int)
                        l = sim_pattern.data['l'].astype(int)
                        sim_hkl = list(zip(h, k, l))
                    else:
                        sim_hkl = [[0, 0, 0]] * len(sim_qx_cart)

                    if 'intensity' in fnames:
                        sim_I = np.asarray(sim_pattern.data['intensity'], dtype=np.float64)

                if sim_qx_cart is not None and len(sim_qx_cart) > 0:
                    if sim_I is None:
                        sim_I = np.ones(len(sim_qx_cart), dtype=np.float64)

                    candidates.append({
                        'rank': idx + 1,
                        'zone_axis': uvw,
                        'correlation': corr_val,
                        'in_plane_angle': in_plane_rot,
                        'matrix': R_mat,
                        'sim_spots': list(zip(sim_qx_cart, sim_qy_cart, sim_hkl, sim_I))
                    })

        return candidates


def export_strain_maps_to_file(filepath: str, strain_maps: dict, valid_mask: np.ndarray = None, metadata: dict = None):
    map_keys = ['eps_xx', 'eps_yy', 'eps_xy', 'omega']
    data_dict = {}
    for k in map_keys:
        arr = strain_maps.get(k, None)
        if arr is None:
            raise ValueError(f"Strain map '{k}' is missing or empty. Please run 'Map 2D Strain' first.")
        arr_copy = np.array(arr, dtype=np.float32, copy=True)
        if valid_mask is not None:
            arr_copy[~valid_mask] = np.nan
        data_dict[k] = arr_copy

    fn_lower = filepath.lower()

    if fn_lower.endswith('.npz'):
        save_payload = dict(data_dict)
        if metadata:
            save_payload['metadata'] = json.dumps(metadata, indent=2, default=str)
        np.savez_compressed(filepath, **save_payload)
        return f"Saved NumPy archive: {os.path.basename(filepath)}"

    elif fn_lower.endswith(('.tif', '.tiff')):
        stack = np.stack([data_dict[k] for k in map_keys], axis=0)
        if HAS_TIFFFILE:
            tifffile.imwrite(
                filepath,
                stack,
                photometric='minisblack',
                metadata={'axes': 'CYX', 'ChannelNames': ['eps_xx', 'eps_yy', 'eps_xy', 'omega']}
            )
        else:
            from PIL import Image
            images = [Image.fromarray(data_dict[k]) for k in map_keys]
            images[0].save(filepath, save_all=True, append_images=images[1:])
        return f"Saved 32-bit TIFF stack: {os.path.basename(filepath)}"
    else:
        raise ValueError("Unsupported extension. Please choose .npz or .tif/.tiff.")


# ==============================================================================
# Secondary Dialogs and Windows
# ==============================================================================
class StrainMapProgressDialog(ctk.CTkToplevel):
    def __init__(self, parent, total_pixels, on_cancel_cb):
        super().__init__(parent)
        self.title("Batch Mapping Progress")
        self.geometry("380x180")
        self.resizable(False, False)
        self.protocol("WM_DELETE_WINDOW", self.on_cancel)

        self.total = max(1, total_pixels)
        self.on_cancel_cb = on_cancel_cb

        f = ctk.CTkFrame(self, corner_radius=10)
        f.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        self.lbl_info = ctk.CTkLabel(f, text="Computing real-space maps...", font=("Arial", 12, "bold"))
        self.lbl_info.pack(anchor='w', padx=10, pady=(6, 4))

        self.pbar = ctk.CTkProgressBar(f, width=320)
        self.pbar.set(0.0)
        self.pbar.pack(padx=10, pady=8)

        self.lbl_status = ctk.CTkLabel(f, text="Pixel: 0 / 0 (0%)", font=("Courier", 11))
        self.lbl_status.pack(padx=10, pady=2)

        self.btn_cancel = ctk.CTkButton(
            f, text="Cancel", width=80, 
            fg_color="#c62828", hover_color="#8e0000", 
            command=self.on_cancel
        )
        self.btn_cancel.pack(pady=(8, 4))

        self.lift()
        self.attributes('-topmost', True)

    def update_progress(self, current, count_str):
        fraction = min(1.0, current / self.total)
        self.pbar.set(fraction)
        self.lbl_status.configure(text=count_str)

    def on_cancel(self):
        if self.on_cancel_cb:
            self.on_cancel_cb()
        self.destroy()

        
class StrainMappingWindow(ctk.CTkToplevel):
    def __init__(self, parent, state, export_callback):
        super().__init__(parent)
        self.title("Real-Space Strain & Deformation Maps")
        self.geometry("1100x820")
        self.minsize(800, 600)

        self.state = state
        self.export_callback = export_callback

        f_top = ctk.CTkFrame(self, height=45, corner_radius=0)
        f_top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(8, 4))

        self.var_mask_vac = tk.BooleanVar(value=True)
        self.chk_mask = ctk.CTkCheckBox(
            f_top, text="Mask Vacuum/Amorphous", 
            variable=self.var_mask_vac, 
            command=self.update_maps,
            font=("Arial", 11)
        )
        self.chk_mask.pack(side=tk.LEFT, padx=(6, 4))

        ctk.CTkLabel(f_top, text="Cutoff (% max):", font=("Arial", 11)).pack(side=tk.LEFT, padx=(6, 2))
        self.var_thresh_pct = ctk.StringVar(value="5.0")
        self.entry_thresh = ctk.CTkEntry(f_top, textvariable=self.var_thresh_pct, width=45, font=("Arial", 11))
        self.entry_thresh.pack(side=tk.LEFT, padx=2)
        self.var_thresh_pct.trace_add("write", lambda *a: self.update_maps())

        ctk.CTkLabel(f_top, text="Cmap:", font=("Arial", 11)).pack(side=tk.LEFT, padx=(12, 2))
        self.var_cmap = ctk.StringVar(value="coolwarm")
        cb_cmap = ctk.CTkOptionMenu(
            f_top, variable=self.var_cmap, 
            values=["coolwarm", "turbo", "viridis", "inferno", "cividis", "bwr", "RdBu_r"], 
            width=95, command=lambda c: self.update_maps()
        )
        cb_cmap.pack(side=tk.LEFT, padx=2)

        ctk.CTkLabel(f_top, text="Clip (%):", font=("Arial", 11)).pack(side=tk.LEFT, padx=(10, 2))
        self.var_clip = tk.DoubleVar(value=98.0)
        self.slider_clip = ctk.CTkSlider(f_top, from_=80.0, to=100.0, number_of_steps=40, width=110, variable=self.var_clip, command=lambda v: self.update_maps())
        self.slider_clip.pack(side=tk.LEFT, padx=2)

        self.btn_export = ctk.CTkButton(
            f_top, text="💾 Export Maps (.npz / .tif)", 
            fg_color="#00695c", hover_color="#004d40", 
            width=190, command=self._trigger_export
        )
        self.btn_export.pack(side=tk.RIGHT, padx=6)

        self.fig, self.axs = plt.subplots(2, 2, figsize=(9.5, 7.5))
        self.fig.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.92, hspace=0.28, wspace=0.32)

        self.canvas_mpl = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas_mpl.draw()
        self.canvas_mpl.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=4)

        self.cbars = [None, None, None, None]
        self.ims = [None, None, None, None]
        self.init_plots()
        self.update_maps()

    def init_plots(self):
        titles = [
            r"Normal Strain $\varepsilon_{xx}$ (%)",
            r"Normal Strain $\varepsilon_{yy}$ (%)",
            r"Shear Strain $\varepsilon_{xy}$ (%)",
            r"Rigid Rotation $\omega$ (deg)"
        ]
        scany, scanx = self.state['scany'], self.state['scanx']
        extent = [0, scanx * self.state['re_samp'], 0, scany * self.state['re_samp']]

        for i, ax in enumerate(self.axs.ravel()):
            ax.clear()
            im = ax.imshow(np.zeros((scany, scanx)), cmap=self.var_cmap.get(), extent=extent, origin=self.state['left_origin'])
            ax.set_title(titles[i], fontsize=11, fontweight='bold', pad=8)
            ax.set_xlabel("x (nm)", fontsize=9)
            ax.set_ylabel("y (nm)", fontsize=9)
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="4%", pad=0.08)
            cbar = self.fig.colorbar(im, cax=cax)
            cbar.ax.tick_params(labelsize=8)
            self.ims[i] = im
            self.cbars[i] = cbar

    def update_maps(self):
        maps = self.state.get('strain_maps', {})
        if maps.get('eps_xx') is None:
            return

        mask = None
        if self.var_mask_vac.get() and self.state['myset'] is not None:
            try: thresh_pct = float(self.var_thresh_pct.get())
            except Exception: thresh_pct = 5.0
            if self.state.get('total_intensity_map') is None:
                self.state['total_intensity_map'] = np.sum(self.state['myset'].data, axis=(2, 3)).astype(np.float64)
            cutoff_val = (thresh_pct / 100.0) * np.max(self.state['total_intensity_map'])
            mask = self.state['total_intensity_map'] < cutoff_val

        map_keys = ['eps_xx', 'eps_yy', 'eps_xy', 'omega']
        clip_pct = self.var_clip.get()

        for i, key in enumerate(map_keys):
            data = maps.get(key)
            if data is None: continue
            
            mult = 100.0 if key != 'omega' else (180.0 / np.pi)
            disp_data = data * mult
            if mask is not None:
                disp_data = disp_data.copy()
                disp_data[mask] = np.nan

            valid_vals = disp_data[~np.isnan(disp_data)]
            if valid_vals.size > 0:
                vmax = float(np.percentile(np.abs(valid_vals), clip_pct))
                if vmax < 1e-4: vmax = 0.5
                vmin = -vmax
            else:
                vmin, vmax = -1.0, 1.0

            im = self.ims[i]
            cbar = self.cbars[i]
            cmap_obj = plt.get_cmap(self.var_cmap.get()).copy()
            cmap_obj.set_bad(color='black', alpha=0.0)
            
            im.set_cmap(cmap_obj)
            im.set_clim(vmin, vmax)
            im.set_data(disp_data)

            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            cbar.mappable.set_norm(norm)
            cbar.mappable.set_clim(vmin, vmax)
            cbar.update_normal(im)

        self.canvas_mpl.draw_idle()

    def _trigger_export(self):
        try:
            thresh_pct = float(self.var_thresh_pct.get())
        except Exception:
            thresh_pct = 5.0
        self.export_callback(self.var_mask_vac.get(), thresh_pct)
        

class PowderPlotDialog(ctk.CTkToplevel):
    """
    Virtual Powder Pattern / Peak Distance Histogram Dialog.
    Aggregates radial distances q with center refinement, distortion correction,
    linearized intensities, dual axes (q vs d-spacing), and weak-peak filtering.
    """
    def __init__(self, parent, framepeaklist, rec_samp=1.0, distortion_matrix=None, on_range_select_cb=None):
        super().__init__(parent)
        self.title("Powder Plot - Bragg Peak Distance Histogram")
        self.geometry("920x680")
        self.minsize(760, 520)

        self.peak_list = framepeaklist
        self.rec_samp = float(rec_samp)
        self.M = np.eye(2, dtype=np.float64) if distortion_matrix is None else np.asarray(distortion_matrix, dtype=np.float64)
        self.on_range_select_cb = on_range_select_cb

        # Aggregate, refine center, and apply distortion matrix
        all_q = []
        all_intensity = []

        for frame in self.peak_list:
            x_arr = frame.get('x', np.array([]))
            y_arr = frame.get('y', np.array([]))
            i_arr = frame.get('intensity', frame.get('raw_mass', frame.get('mass', np.array([]))))

            if len(x_arr) < 2:  # Need at least central beam + 1 diffracted peak
                continue

            # Identify strongest peak as central beam origin
            center_idx = int(np.argmax(i_arr))
            cx, cy = x_arr[center_idx], y_arr[center_idx]

            # Shift origin relative to refined center
            rel_x = x_arr - cx
            rel_y = y_arr - cy

            # Exclude central peak itself
            mask = np.ones(len(x_arr), dtype=bool)
            mask[center_idx] = False

            rel_x_diff = rel_x[mask]
            rel_y_diff = rel_y[mask]
            rel_i_diff = i_arr[mask]

            if len(rel_x_diff) == 0:
                continue

            # Correct peak coordinates using reciprocal distortion matrix M
            coords = np.column_stack([rel_x_diff, rel_y_diff])
            corr_coords = (self.M @ coords.T).T

            # Calculate physical reciprocal distances: q = rec_samp * sqrt(x_corr^2 + y_corr^2)
            q_physical = self.rec_samp * np.hypot(corr_coords[:, 0], corr_coords[:, 1])

            all_q.append(q_physical)
            all_intensity.append(rel_i_diff)

        if all_q:
            self.q_vals = np.concatenate(all_q)
            self.intensity_vals = np.concatenate(all_intensity)
        else:
            self.q_vals = np.array([], dtype=np.float64)
            self.intensity_vals = np.array([], dtype=np.float64)

        # Control Bar Top
        f_top = ctk.CTkFrame(self, height=45, corner_radius=0)
        f_top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(8, 4))

        ctk.CTkLabel(f_top, text="Weighting:", font=("Arial", 11, "bold")).pack(side=tk.LEFT, padx=(10, 4))
        self.var_weight = ctk.StringVar(value="Linear Intensity")
        cb_weight = ctk.CTkOptionMenu(
            f_top, variable=self.var_weight,
            values=["Linear Intensity", "Peak Frequency"],
            width=150, command=lambda c: self.update_plot()
        )
        cb_weight.pack(side=tk.LEFT, padx=4)

        ctk.CTkLabel(f_top, text="Bins:", font=("Arial", 11, "bold")).pack(side=tk.LEFT, padx=(12, 4))
        self.var_bins = tk.IntVar(value=150)
        self.lbl_bins = ctk.CTkLabel(f_top, text=str(self.var_bins.get()), font=("Courier", 11, "bold"), width=32)
        self.lbl_bins.pack(side=tk.LEFT, padx=2)

        def _on_bin_slider(v):
            val = int(round(float(v)))
            self.var_bins.set(val)
            self.lbl_bins.configure(text=str(val))
            self.update_plot()

        slider_bins = ctk.CTkSlider(
            f_top, from_=30, to=500, number_of_steps=94,
            command=_on_bin_slider, width=110
        )
        slider_bins.set(self.var_bins.get())
        slider_bins.pack(side=tk.LEFT, padx=4)

        # Weak peak cut-off slider
        ctk.CTkLabel(f_top, text="Cut-off (%):", font=("Arial", 11, "bold"), text_color="#ffca28").pack(side=tk.LEFT, padx=(12, 4))
        self.var_cutoff = tk.DoubleVar(value=0.0)
        self.lbl_cutoff = ctk.CTkLabel(f_top, text="0%", font=("Courier", 11, "bold"), width=36)
        self.lbl_cutoff.pack(side=tk.LEFT, padx=2)

        def _on_cutoff_slider(v):
            pct = int(round(float(v)))
            self.lbl_cutoff.configure(text=f"{pct}%")
            self.update_plot()

        slider_cutoff = ctk.CTkSlider(
            f_top, from_=0.0, to=90.0, number_of_steps=90,
            variable=self.var_cutoff, command=_on_cutoff_slider, width=110
        )
        slider_cutoff.pack(side=tk.LEFT, padx=4)

        self.btn_export_csv = ctk.CTkButton(
            f_top, text="💾 Export CSV", width=105,
            fg_color="#00695c", hover_color="#004d40",
            command=self.export_csv
        )
        self.btn_export_csv.pack(side=tk.RIGHT, padx=6)

        # Matplotlib Figure
        self.fig, self.ax = plt.subplots(figsize=(8.0, 5.0))
        self.fig.patch.set_facecolor('#1a1a1a')
        self.ax.set_facecolor('#121212')
        self.fig.subplots_adjust(top=0.88, bottom=0.12, left=0.10, right=0.94)
        self.secax = None

        self.canvas_mpl = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas_widget = self.canvas_mpl.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=4)

        # Bottom Frame
        f_bot = ctk.CTkFrame(self, height=40, corner_radius=6)
        f_bot.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=(2, 8))

        self.lbl_info = ctk.CTkLabel(
            f_bot,
            text=f"Total Bragg Peaks: {len(self.q_vals):,} (Refined & Distortion Corrected)",
            font=("Arial", 11)
        )
        self.lbl_info.pack(side=tk.LEFT, padx=12, pady=6)

        self.btn_apply_interval = ctk.CTkButton(
            f_bot, text="Set Range to Map", width=140,
            fg_color="#6a1b9a", hover_color="#4a148c",
            command=self._apply_range_callback
        )
        self.btn_apply_interval.pack(side=tk.RIGHT, padx=10, pady=6)

        self.var_q_max = ctk.StringVar(value="")
        self.entry_q_max = ctk.CTkEntry(f_bot, textvariable=self.var_q_max, width=60, font=("Courier", 11))
        self.entry_q_max.pack(side=tk.RIGHT, padx=2)
        ctk.CTkLabel(f_bot, text="q_max:", font=("Arial", 11)).pack(side=tk.RIGHT, padx=(6, 2))

        self.var_q_min = ctk.StringVar(value="")
        self.entry_q_min = ctk.CTkEntry(f_bot, textvariable=self.var_q_min, width=60, font=("Courier", 11))
        self.entry_q_min.pack(side=tk.RIGHT, padx=2)
        ctk.CTkLabel(f_bot, text="q_min:", font=("Arial", 11)).pack(side=tk.RIGHT, padx=(6, 2))

        self.hist_bin_centers = np.array([])
        self.hist_values = np.array([])

        self.update_plot()

    def update_plot(self):
        if self.secax is not None:
            try:
                self.secax.remove()
            except Exception:
                pass
            self.secax = None

        self.ax.clear()
        self.ax.tick_params(colors='white', labelsize=9)
        self.ax.set_facecolor('#121212')

        valid_mask = self.q_vals > 1e-4
        if not np.any(valid_mask):
            self.ax.text(0.5, 0.5, "No diffracted peaks detected.\nRun 'Map All Peaks (ROI)' first.",
                         transform=self.ax.transAxes, color='white', ha='center', va='center', fontsize=12)
            self.canvas_mpl.draw_idle()
            return

        filt_q = self.q_vals[valid_mask]
        filt_int = self.intensity_vals[valid_mask]

        # Apply weak-peak cut-off threshold
        cutoff_pct = float(self.var_cutoff.get())
        if cutoff_pct > 0.0 and len(filt_int) > 0:
            thresh_val = np.percentile(filt_int, cutoff_pct)
            surv_mask = filt_int >= thresh_val
            filt_q = filt_q[surv_mask]
            filt_int = filt_int[surv_mask]

        if len(filt_q) == 0:
            self.ax.text(0.5, 0.5, "All peaks cut off by current threshold slider.",
                         transform=self.ax.transAxes, color='white', ha='center', va='center', fontsize=11)
            self.canvas_mpl.draw_idle()
            return

        n_bins = self.var_bins.get()
        use_linear = (self.var_weight.get() == "Linear Intensity")
        weights = filt_int if use_linear else None

        min_q = float(np.min(filt_q))
        max_q = float(np.max(filt_q))
        if np.isclose(min_q, max_q):
            min_q = max(1e-3, min_q * 0.9)
            max_q = max_q * 1.1

        counts, bin_edges = np.histogram(filt_q, bins=n_bins, range=(min_q, max_q), weights=weights)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_width = bin_edges[1] - bin_edges[0]

        self.hist_bin_centers = bin_centers
        self.hist_values = counts

        self.ax.bar(
            bin_centers, counts, width=bin_width * 0.95,
            color="#00e5ff", edgecolor="#0097a7", alpha=0.75, align='center'
        )

        ylabel = "Linear Integrated Intensity (a.u.)" if use_linear else "Peak Frequency (Count)"
        self.ax.set_ylabel(ylabel, color='white', fontsize=10, labelpad=6)
        self.ax.set_xlabel(r"Reciprocal Distance $q$ ($1/\mathrm{nm}$)", color='white', fontsize=10, labelpad=6)
        self.ax.set_title("Virtual Powder Pattern (Distortion-Corrected & Centered)", color='white', fontsize=11, fontweight='bold', pad=12)
        self.ax.grid(True, linestyle=':', color='#37474f', alpha=0.6)

        # Pin parent x-axis strictly positive
        q_left = max(1e-3, min_q - bin_width)
        q_right = max_q + bin_width
        self.ax.set_xlim(left=q_left, right=q_right)

        # Dual Axes: Reciprocal Distance (q) & Real Space Distance (d = 1/q)
        def q2d(x):
            x_arr = np.clip(np.asarray(x, dtype=np.float64), 1e-4, 1e5)
            return 1.0 / x_arr

        def d2q(d):
            d_arr = np.clip(np.asarray(d, dtype=np.float64), 1e-4, 1e5)
            return 1.0 / d_arr

        self.secax = self.ax.secondary_xaxis('top', functions=(q2d, d2q))
        self.secax.set_xlabel(r"Real Space Lattice Spacing $d$ ($\mathrm{nm}$)", color='#80cbc4', fontsize=9, labelpad=5)
        self.secax.tick_params(colors='#80cbc4', labelsize=8)

        # Initialize input bounds
        if not self.var_q_min.get():
            self.var_q_min.set(f"{float(np.percentile(filt_q, 25)):.2f}")
            self.var_q_max.set(f"{float(np.percentile(filt_q, 75)):.2f}")

        self.lbl_info.configure(text=f"Retained Peaks: {len(filt_q):,} / {len(self.q_vals):,} (Cut-off: {int(cutoff_pct)}%)")
        self.canvas_mpl.draw_idle()

    def _apply_range_callback(self):
        try:
            q_min = float(self.var_q_min.get())
            q_max = float(self.var_q_max.get())
            if q_min >= q_max:
                raise ValueError("q_min must be strictly less than q_max.")
        except ValueError as e:
            messagebox.showerror("Invalid Range", str(e), parent=self)
            return

        for patch in list(self.ax.patches):
            if isinstance(patch, Rectangle) and patch.get_facecolor() == mcolors.to_rgba('#ff1744', 0.25):
                patch.remove()

        ylim = self.ax.get_ylim()
        span_rect = Rectangle(
            (q_min, ylim[0]), q_max - q_min, ylim[1] - ylim[0],
            facecolor='#ff1744', alpha=0.25, edgecolor='#ff1744', linestyle='--', linewidth=1.5, zorder=10
        )
        self.ax.add_patch(span_rect)
        self.canvas_mpl.draw_idle()

        if self.on_range_select_cb:
            self.on_range_select_cb(q_min, q_max)

    def export_csv(self):
        if len(self.hist_bin_centers) == 0:
            messagebox.showwarning("No Data", "No histogram values to export.", parent=self)
            return

        fn = filedialog.asksaveasfilename(
            parent=self,
            defaultextension=".csv",
            filetypes=[("CSV Spreadsheet", "*.csv"), ("All Files", "*.*")]
        )
        if fn:
            try:
                d_spacing = np.divide(
                    1.0, self.hist_bin_centers,
                    out=np.zeros_like(self.hist_bin_centers),
                    where=self.hist_bin_centers > 0
                )
                data_out = np.column_stack([self.hist_bin_centers, d_spacing, self.hist_values])
                header = "q_1_per_nm,d_spacing_nm,intensity"
                np.savetxt(fn, data_out, delimiter=",", header=header, comments="")
                messagebox.showinfo("Export Complete", f"Saved histogram to:\n{os.path.basename(fn)}", parent=self)
            except Exception as e:
                messagebox.showerror("Export Error", str(e), parent=self)

from matplotlib.widgets import SpanSelector

class PowderPlotDialog(ctk.CTkToplevel):
    """
    Virtual Powder Pattern / Peak Distance Histogram Dialog.
    Features:
    - SpanSelector for interactive mouse drag-selection of [q_min, q_max]
    - Live crosshair cursor measuring q (1/nm) and d-spacing (nm / Å)
    - Default bins set to 2x detector dimension
    - Linearized, centered, and distortion-corrected peak accumulation
    """
    def __init__(self, parent, framepeaklist, rec_samp=1.0, det_shape=(128, 128),
                 distortion_matrix=None, on_range_select_cb=None):
        super().__init__(parent)
        self.title("Powder Plot - Bragg Peak Distance Histogram")
        self.geometry("940x700")
        self.minsize(780, 540)

        self.peak_list = framepeaklist
        self.rec_samp = float(rec_samp)
        self.det_shape = det_shape  # (dety, detx)
        self.M = np.eye(2, dtype=np.float64) if distortion_matrix is None else np.asarray(distortion_matrix, dtype=np.float64)
        self.on_range_select_cb = on_range_select_cb

        # Aggregate, refine center, and apply distortion matrix
        all_q = []
        all_intensity = []

        for frame in self.peak_list:
            x_arr = frame.get('x', np.array([]))
            y_arr = frame.get('y', np.array([]))
            i_arr = frame.get('intensity', frame.get('raw_mass', frame.get('mass', np.array([]))))

            if len(x_arr) < 2:
                continue

            center_idx = int(np.argmax(i_arr))
            cx, cy = x_arr[center_idx], y_arr[center_idx]

            mask = np.ones(len(x_arr), dtype=bool)
            mask[center_idx] = False

            rel_x_diff = x_arr[mask] - cx
            rel_y_diff = y_arr[mask] - cy
            rel_i_diff = i_arr[mask]

            if len(rel_x_diff) == 0:
                continue

            coords = np.column_stack([rel_x_diff, rel_y_diff])
            corr_coords = (self.M @ coords.T).T

            q_physical = self.rec_samp * np.hypot(corr_coords[:, 0], corr_coords[:, 1])

            all_q.append(q_physical)
            all_intensity.append(rel_i_diff)

        if all_q:
            self.q_vals = np.concatenate(all_q)
            self.intensity_vals = np.concatenate(all_intensity)
        else:
            self.q_vals = np.array([], dtype=np.float64)
            self.intensity_vals = np.array([], dtype=np.float64)

        # Default bins = 2 * detector dimension
        det_dim = max(int(self.det_shape[0]), int(self.det_shape[1])) if self.det_shape else 128
        default_bins = max(50, int(2 * det_dim))

        # Control Bar Top
        f_top = ctk.CTkFrame(self, height=45, corner_radius=0)
        f_top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(8, 4))

        ctk.CTkLabel(f_top, text="Weighting:", font=("Arial", 11, "bold")).pack(side=tk.LEFT, padx=(10, 4))
        self.var_weight = ctk.StringVar(value="Linear Intensity")
        cb_weight = ctk.CTkOptionMenu(
            f_top, variable=self.var_weight,
            values=["Linear Intensity", "Peak Frequency"],
            width=150, command=lambda c: self.update_plot()
        )
        cb_weight.pack(side=tk.LEFT, padx=4)

        ctk.CTkLabel(f_top, text="Bins:", font=("Arial", 11, "bold")).pack(side=tk.LEFT, padx=(12, 4))
        self.var_bins = tk.IntVar(value=default_bins)
        self.lbl_bins = ctk.CTkLabel(f_top, text=str(default_bins), font=("Courier", 11, "bold"), width=36)
        self.lbl_bins.pack(side=tk.LEFT, padx=2)

        def _on_bin_slider(v):
            val = int(round(float(v)))
            self.var_bins.set(val)
            self.lbl_bins.configure(text=str(val))
            self.update_plot()

        slider_max = max(1000, int(default_bins * 2.5))
        slider_bins = ctk.CTkSlider(
            f_top, from_=30, to=slider_max, number_of_steps=slider_max - 30,
            command=_on_bin_slider, width=120
        )
        slider_bins.set(default_bins)
        slider_bins.pack(side=tk.LEFT, padx=4)

        ctk.CTkLabel(f_top, text="Cut-off (%):", font=("Arial", 11, "bold"), text_color="#ffca28").pack(side=tk.LEFT, padx=(12, 4))
        self.var_cutoff = tk.DoubleVar(value=0.0)
        self.lbl_cutoff = ctk.CTkLabel(f_top, text="0%", font=("Courier", 11, "bold"), width=36)
        self.lbl_cutoff.pack(side=tk.LEFT, padx=2)

        def _on_cutoff_slider(v):
            pct = int(round(float(v)))
            self.lbl_cutoff.configure(text=f"{pct}%")
            self.update_plot()

        slider_cutoff = ctk.CTkSlider(
            f_top, from_=0.0, to=90.0, number_of_steps=90,
            variable=self.var_cutoff, command=_on_cutoff_slider, width=110
        )
        slider_cutoff.pack(side=tk.LEFT, padx=4)

        self.btn_export_csv = ctk.CTkButton(
            f_top, text="💾 Export CSV", width=105,
            fg_color="#00695c", hover_color="#004d40",
            command=self.export_csv
        )
        self.btn_export_csv.pack(side=tk.RIGHT, padx=6)

        # Matplotlib Figure
        self.fig, self.ax = plt.subplots(figsize=(8.5, 5.2))
        self.fig.patch.set_facecolor('#1a1a1a')
        self.ax.set_facecolor('#121212')
        self.fig.subplots_adjust(top=0.88, bottom=0.12, left=0.10, right=0.94)
        self.secax = None
        self.span_selector = None

        self.canvas_mpl = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas_widget = self.canvas_mpl.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=4)

        # Crosshair cursor lines & info box
        self.ch_vline = self.ax.axvline(np.nan, color='#ffeb3b', linestyle=':', lw=1.2, alpha=0.9, zorder=15)
        self.ch_hline = self.ax.axhline(np.nan, color='#ffeb3b', linestyle=':', lw=1.2, alpha=0.9, zorder=15)
        self.ch_hud = self.ax.text(
            0.02, 0.95, "", transform=self.ax.transAxes, color='white',
            fontsize=9, fontfamily="monospace", zorder=20,
            bbox=dict(boxstyle="round,pad=0.4", fc="#1a1a1a", ec="#ffeb3b", alpha=0.85)
        )
        self.ch_hud.set_visible(False)

        # Bottom Frame
        f_bot = ctk.CTkFrame(self, height=40, corner_radius=6)
        f_bot.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=(2, 8))

        self.lbl_info = ctk.CTkLabel(
            f_bot,
            text=f"Total Bragg Peaks: {len(self.q_vals):,} (Bins: {default_bins} = 2×{det_dim}px)",
            font=("Arial", 11)
        )
        self.lbl_info.pack(side=tk.LEFT, padx=12, pady=6)

        self.btn_apply_interval = ctk.CTkButton(
            f_bot, text="Set Range to Map", width=140,
            fg_color="#6a1b9a", hover_color="#4a148c",
            command=self._apply_range_callback
        )
        self.btn_apply_interval.pack(side=tk.RIGHT, padx=10, pady=6)

        self.var_q_max = ctk.StringVar(value="")
        self.entry_q_max = ctk.CTkEntry(f_bot, textvariable=self.var_q_max, width=65, font=("Courier", 11))
        self.entry_q_max.pack(side=tk.RIGHT, padx=2)
        ctk.CTkLabel(f_bot, text="q_max:", font=("Arial", 11)).pack(side=tk.RIGHT, padx=(6, 2))

        self.var_q_min = ctk.StringVar(value="")
        self.entry_q_min = ctk.CTkEntry(f_bot, textvariable=self.var_q_min, width=65, font=("Courier", 11))
        self.entry_q_min.pack(side=tk.RIGHT, padx=2)
        ctk.CTkLabel(f_bot, text="q_min:", font=("Arial", 11)).pack(side=tk.RIGHT, padx=(6, 2))

        self.hist_bin_centers = np.array([])
        self.hist_values = np.array([])

        # Connect mouse motion listener for crosshair
        self.canvas_mpl.mpl_connect("motion_notify_event", self.on_mouse_move)

        self.update_plot()

    def update_plot(self):
        if self.secax is not None:
            try:
                self.secax.remove()
            except Exception:
                pass
            self.secax = None

        self.ax.clear()
        self.ax.tick_params(colors='white', labelsize=9)
        self.ax.set_facecolor('#121212')

        # Re-attach crosshair artists to cleared axes
        self.ch_vline = self.ax.axvline(np.nan, color='#ffeb3b', linestyle=':', lw=1.2, alpha=0.9, zorder=15)
        self.ch_hline = self.ax.axhline(np.nan, color='#ffeb3b', linestyle=':', lw=1.2, alpha=0.9, zorder=15)
        self.ch_hud = self.ax.text(
            0.02, 0.95, "", transform=self.ax.transAxes, color='white',
            fontsize=9, fontfamily="monospace", zorder=20,
            bbox=dict(boxstyle="round,pad=0.4", fc="#1a1a1a", ec="#ffeb3b", alpha=0.85)
        )
        self.ch_hud.set_visible(False)

        valid_mask = self.q_vals > 1e-4
        if not np.any(valid_mask):
            self.ax.text(0.5, 0.5, "No diffracted peaks detected.\nRun 'Map All Peaks (ROI)' first.",
                         transform=self.ax.transAxes, color='white', ha='center', va='center', fontsize=12)
            self.canvas_mpl.draw_idle()
            return

        filt_q = self.q_vals[valid_mask]
        filt_int = self.intensity_vals[valid_mask]

        cutoff_pct = float(self.var_cutoff.get())
        if cutoff_pct > 0.0 and len(filt_int) > 0:
            thresh_val = np.percentile(filt_int, cutoff_pct)
            surv_mask = filt_int >= thresh_val
            filt_q = filt_q[surv_mask]
            filt_int = filt_int[surv_mask]

        if len(filt_q) == 0:
            self.ax.text(0.5, 0.5, "All peaks cut off by current threshold slider.",
                         transform=self.ax.transAxes, color='white', ha='center', va='center', fontsize=11)
            self.canvas_mpl.draw_idle()
            return

        n_bins = self.var_bins.get()
        use_linear = (self.var_weight.get() == "Linear Intensity")
        weights = filt_int if use_linear else None

        min_q = float(np.min(filt_q))
        max_q = float(np.max(filt_q))
        if np.isclose(min_q, max_q):
            min_q = max(1e-3, min_q * 0.9)
            max_q = max_q * 1.1

        counts, bin_edges = np.histogram(filt_q, bins=n_bins, range=(min_q, max_q), weights=weights)
        bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
        bin_width = bin_edges[1] - bin_edges[0]

        self.hist_bin_centers = bin_centers
        self.hist_values = counts

        self.ax.bar(
            bin_centers, counts, width=bin_width * 0.95,
            color="#00e5ff", edgecolor="#0097a7", alpha=0.75, align='center', zorder=5
        )

        ylabel = "Linear Integrated Intensity (a.u.)" if use_linear else "Peak Frequency (Count)"
        self.ax.set_ylabel(ylabel, color='white', fontsize=10, labelpad=6)
        self.ax.set_xlabel(r"Reciprocal Distance $q$ ($1/\mathrm{nm}$)", color='white', fontsize=10, labelpad=6)
        self.ax.set_title("Virtual Powder Pattern (Distortion-Corrected & Centered)", color='white', fontsize=11, fontweight='bold', pad=12)
        self.ax.grid(True, linestyle=':', color='#37474f', alpha=0.6, zorder=1)

        q_left = max(1e-3, min_q - bin_width)
        q_right = max_q + bin_width
        self.ax.set_xlim(left=q_left, right=q_right)

        # Dual Axes: Reciprocal Distance q (1/nm) & Real Space Spacing d (nm)
        def q2d(x):
            x_arr = np.clip(np.asarray(x, dtype=np.float64), 1e-4, 1e5)
            return 1.0 / x_arr

        def d2q(d):
            d_arr = np.clip(np.asarray(d, dtype=np.float64), 1e-4, 1e5)
            return 1.0 / d_arr

        self.secax = self.ax.secondary_xaxis('top', functions=(q2d, d2q))
        self.secax.set_xlabel(r"Real Space Lattice Spacing $d$ ($\mathrm{nm}$)", color='#80cbc4', fontsize=9, labelpad=5)
        self.secax.tick_params(colors='#80cbc4', labelsize=8)

        # Initialize input bounds
        if not self.var_q_min.get():
            self.var_q_min.set(f"{float(np.percentile(filt_q, 25)):.3f}")
            self.var_q_max.set(f"{float(np.percentile(filt_q, 75)):.3f}")

        # Setup SpanSelector for mouse drag-selection
        self.span_selector = SpanSelector(
            self.ax,
            onselect=self.on_span_select,
            direction="horizontal",
            useblit=True,
            props=dict(alpha=0.3, facecolor="#ff1744", edgecolor="#ff1744", linestyle="--"),
            interactive=True,
            drag_from_anywhere=True
        )

        self.lbl_info.configure(text=f"Retained Peaks: {len(filt_q):,} / {len(self.q_vals):,} (Cut-off: {int(cutoff_pct)}%)")
        self.canvas_mpl.draw_idle()

    def on_span_select(self, qmin, qmax):
        """Callback fired when dragging the horizontal SpanSelector."""
        qmin_f = min(float(qmin), float(qmax))
        qmax_f = max(float(qmin), float(qmax))

        self.var_q_min.set(f"{qmin_f:.3f}")
        self.var_q_max.set(f"{qmax_f:.3f}")

    def on_mouse_move(self, event):
        """Crosshair cursor measurement readout for q (1/nm) and d (nm / Å)."""
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            self.ch_vline.set_visible(False)
            self.ch_hline.set_visible(False)
            self.ch_hud.set_visible(False)
            self.canvas_mpl.draw_idle()
            return

        qx = float(event.xdata)
        qy = float(event.ydata)

        if qx <= 1e-4:
            return

        d_nm = 1.0 / qx
        d_angstrom = d_nm * 10.0

        self.ch_vline.set_xdata([qx, qx])
        self.ch_hline.set_ydata([qy, qy])
        self.ch_vline.set_visible(True)
        self.ch_hline.set_visible(True)

        hud_text = (
            f"q: {qx:.4f} 1/nm\n"
            f"d: {d_nm:.4f} nm ({d_angstrom:.3f} Å)\n"
            f"Y: {qy:.2e}"
        )
        self.ch_hud.set_text(hud_text)
        self.ch_hud.set_visible(True)
        self.canvas_mpl.draw_idle()

    def _apply_range_callback(self):
        try:
            q_min = float(self.var_q_min.get())
            q_max = float(self.var_q_max.get())
            if q_min >= q_max:
                raise ValueError("q_min must be strictly less than q_max.")
        except ValueError as e:
            messagebox.showerror("Invalid Range", str(e), parent=self)
            return

        if self.on_range_select_cb:
            self.on_range_select_cb(q_min, q_max)

    def export_csv(self):
        if len(self.hist_bin_centers) == 0:
            messagebox.showwarning("No Data", "No histogram values to export.", parent=self)
            return

        fn = filedialog.asksaveasfilename(
            parent=self,
            defaultextension=".csv",
            filetypes=[("CSV Spreadsheet", "*.csv"), ("All Files", "*.*")]
        )
        if fn:
            try:
                d_spacing = np.divide(
                    1.0, self.hist_bin_centers,
                    out=np.zeros_like(self.hist_bin_centers),
                    where=self.hist_bin_centers > 0
                )
                data_out = np.column_stack([self.hist_bin_centers, d_spacing, self.hist_values])
                header = "q_1_per_nm,d_spacing_nm,intensity"
                np.savetxt(fn, data_out, delimiter=",", header=header, comments="")
                messagebox.showinfo("Export Complete", f"Saved histogram to:\n{os.path.basename(fn)}", parent=self)
            except Exception as e:
                messagebox.showerror("Export Error", str(e), parent=self)

class OrientationMappingWindow(ctk.CTkToplevel):
    def __init__(self, parent, state, export_callback, on_probe_select_callback=None):
        super().__init__(parent)
        self.title("Interactive 2D Orientation & Stereographic Projection Explorer")
        self.geometry("1220x700")
        self.minsize(900, 520)

        self.state = state
        self.export_callback = export_callback
        self.on_probe_select_callback = on_probe_select_callback
        self.crystal = state['acom_mgr'].crystal

        self.orient_maps = state.get('orientation_maps', {})
        self.R_matrices = self.orient_maps.get('R_matrices')
        self.corr_map = self.orient_maps.get('corr')
        self.angle_map = self.orient_maps.get('angle')
        self._last_hover_pos = None

        # Control Bar
        f_top = ctk.CTkFrame(self, height=45, corner_radius=0)
        f_top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(8, 4))

        ctk.CTkLabel(f_top, text="Display Map:", font=("Arial", 11, "bold")).pack(side=tk.LEFT, padx=(10, 4))
        self.var_map_mode = ctk.StringVar(value="Correlation (Q)")
        cb_mode = ctk.CTkOptionMenu(
            f_top, variable=self.var_map_mode,
            values=["Correlation (Q)", "In-Plane Angle (deg)", "Zone Axis ID"],
            width=150, command=lambda c: self.update_spatial_map()
        )
        cb_mode.pack(side=tk.LEFT, padx=4)

        ctk.CTkLabel(f_top, text="Min Q:", font=("Arial", 11, "bold"), text_color="#00e5ff").pack(side=tk.LEFT, padx=(14, 2))
        self.var_q_min = tk.DoubleVar(value=0.20)
        self.lbl_q_val = ctk.CTkLabel(f_top, text=f"{self.var_q_min.get():.2f}", font=("Courier", 11, "bold"), width=36)
        self.lbl_q_val.pack(side=tk.LEFT, padx=2)

        def _on_q_slider(val):
            self.lbl_q_val.configure(text=f"{float(val):.2f}")
            self.update_spatial_map()

        self.slider_q = ctk.CTkSlider(
            f_top, from_=0.0, to=1.0, number_of_steps=100,
            variable=self.var_q_min, command=_on_q_slider, width=120
        )
        self.slider_q.pack(side=tk.LEFT, padx=(2, 6))

        self.var_mask_vac = tk.BooleanVar(value=True)
        self.chk_mask = ctk.CTkCheckBox(
            f_top, text="Mask Vacuum", variable=self.var_mask_vac,
            command=self.update_spatial_map, font=("Arial", 11)
        )
        self.chk_mask.pack(side=tk.LEFT, padx=(10, 4))

        self.btn_export = ctk.CTkButton(
            f_top, text="💾 Export Maps (.npz / .tif)",
            fg_color="#6a1b9a", hover_color="#4a148c",
            width=180, command=self._trigger_export
        )
        self.btn_export.pack(side=tk.RIGHT, padx=6)

        # Matplotlib Layout Setup
        self.fig, (self.ax_map, self.ax_stereo) = plt.subplots(1, 2, figsize=(11.5, 5.5))
        self.fig.patch.set_facecolor('#1a1a1a')
        self.ax_map.set_facecolor('#121212')
        self.ax_stereo.set_facecolor('#121212')
        self.fig.subplots_adjust(top=0.90, bottom=0.10, left=0.08, right=0.92, wspace=0.35)

        self.canvas_mpl = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas_widget = self.canvas_mpl.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=4)

        self.im_map = None
        self.cbar_map = None
        self.cax = None
        self.cursor_marker = None
        self.stereo_spot = None
        self.stereo_text = None

        self.init_plots()
        self.update_spatial_map()

        self.canvas_mpl.mpl_connect("motion_notify_event", self.on_map_hover)
        self.canvas_mpl.mpl_connect("button_press_event", self.on_map_click)

        self.lift()
        self.after(100, self.focus_force)

    def init_plots(self):
        scany, scanx = self.state['scany'], self.state['scanx']
        extent = [0, scanx * self.state['re_samp'], 0, scany * self.state['re_samp']]

        self.ax_map.clear()
        self.im_map = self.ax_map.imshow(
            np.zeros((scany, scanx)), cmap="viridis", extent=extent,
            origin=self.state['left_origin']
        )
        self.ax_map.set_title("Orientation Map", fontsize=11, fontweight='bold', pad=8, color='white')
        self.ax_map.set_xlabel("x (nm)", fontsize=9, color='white')
        self.ax_map.set_ylabel("y (nm)", fontsize=9, color='white')
        self.ax_map.tick_params(colors='white')

        div = make_axes_locatable(self.ax_map)
        self.cax = div.append_axes("right", size="4%", pad=0.08)
        self.cbar_map = self.fig.colorbar(self.im_map, cax=self.cax)
        self.cbar_map.ax.tick_params(colors='white', labelsize=8)

        self.cursor_marker = self.ax_map.plot([], [], 'r+', markersize=12, markeredgewidth=2.0, zorder=10)[0]
        self.setup_stereographic_axes()
        self.canvas_mpl.draw()

    def setup_stereographic_axes(self):
        self.ax_stereo.clear()
        self.ax_stereo.set_title("Stereographic Projection (Beam Axis Z)", fontsize=11, fontweight='bold', pad=8, color='white')
        self.ax_stereo.set_aspect('equal')
        self.ax_stereo.set_xlim(-1.15, 1.15)
        self.ax_stereo.set_ylim(-1.15, 1.15)
        self.ax_stereo.tick_params(colors='#78909c')

        unit_circle = Circle((0, 0), 1.0, edgecolor='#76ff03', facecolor='none', linewidth=1.5, linestyle='-')
        self.ax_stereo.add_patch(unit_circle)

        self.ax_stereo.axhline(0, color='#37474f', linestyle=':', lw=1.0)
        self.ax_stereo.axvline(0, color='#37474f', linestyle=':', lw=1.0)
        inner_ring = Circle((0, 0), np.tan(np.radians(22.5)), edgecolor='#263238', facecolor='none', linestyle=':', lw=0.8)
        self.ax_stereo.add_patch(inner_ring)

        self.ax_stereo.text(1.05, 0.0, "X", color='#76ff03', va='center', fontsize=9, fontweight='bold')
        self.ax_stereo.text(0.0, 1.05, "Y", color='#76ff03', ha='center', fontsize=9, fontweight='bold')
        self.ax_stereo.text(-0.05, -0.05, "Z [001]", color='#90a4ae', fontsize=8)

        self.stereo_spot = self.ax_stereo.plot(
            [], [], marker='o', markersize=10, markerfacecolor='#ff1744',
            markeredgecolor='white', markeredgewidth=1.5, zorder=8
        )[0]

        self.stereo_text = self.ax_stereo.text(
            0.03, 0.05, "Hover or click on the map\nto inspect orientation",
            transform=self.ax_stereo.transAxes, color='white', fontsize=9, fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.4", fc="#121212", ec="#00e5ff", alpha=0.9),
            zorder=9
        )

    def _resolve_coordinates(self, event):
        if event.inaxes != self.ax_map or event.xdata is None or event.ydata is None:
            return None, None
        px = int(np.clip(event.xdata / self.state['re_samp'], 0, self.state['scanx'] - 1))
        if self.state['left_origin'] == 'upper':
            py = int(np.clip(self.state['scany'] - 1 - (event.ydata / self.state['re_samp']), 0, self.state['scany'] - 1))
        else:
            py = int(np.clip(event.ydata / self.state['re_samp'], 0, self.state['scany'] - 1))
        return py, px

    def on_map_hover(self, event):
        py, px = self._resolve_coordinates(event)
        if py is not None and px is not None:
            if getattr(self, '_last_hover_pos', None) == (py, px):
                return
            self._last_hover_pos = (py, px)
            self.update_stereo_projection_at_pixel(py, px)

    def on_map_click(self, event):
        if event.button != 1:
            return
        py, px = self._resolve_coordinates(event)
        if py is not None and px is not None:
            self.update_stereo_projection_at_pixel(py, px)
            if self.on_probe_select_callback is not None:
                self.on_probe_select_callback(py, px)

    def update_spatial_map(self):
        mode = self.var_map_mode.get()
        if self.corr_map is None:
            return

        q_min = float(self.var_q_min.get())
        mask = self.corr_map < q_min

        if self.var_mask_vac.get() and self.state['myset'] is not None:
            if self.state.get('total_intensity_map') is None:
                self.state['total_intensity_map'] = np.sum(self.state['myset'].data, axis=(2, 3)).astype(np.float64)
            cutoff_val = 0.05 * np.max(self.state['total_intensity_map'])
            mask |= (self.state['total_intensity_map'] < cutoff_val)

        if mode == "Correlation (Q)":
            data = np.array(self.corr_map, copy=True)
            cmap_name = "viridis"
            vmin, vmax = 0.0, 1.0
            label = "Correlation Q"
        elif mode == "In-Plane Angle (deg)":
            data = np.array(self.angle_map, copy=True)
            cmap_name = "hsv"
            vmin, vmax = -180.0, 180.0
            label = "Angle θ (°)"
        else:
            data = np.array(self.orient_maps.get('zone_axis_id', self.corr_map), copy=True)
            cmap_name = "turbo"
            valid = data[~np.isnan(data)]
            vmin = float(np.min(valid)) if valid.size > 0 else 0.0
            vmax = float(np.max(valid)) if valid.size > 0 else 1.0
            label = "Zone Axis ID"

        data[mask] = np.nan
        cmap_obj = plt.get_cmap(cmap_name).copy()
        cmap_obj.set_bad(color='black', alpha=0.0)

        self.im_map.set_cmap(cmap_obj)
        self.im_map.set_clim(vmin, vmax)
        self.im_map.set_data(data)

        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        self.cbar_map.mappable.set_norm(norm)
        self.cbar_map.mappable.set_clim(vmin, vmax)
        self.cbar_map.set_label(label, fontsize=9, color='white')
        self.cbar_map.update_normal(self.im_map)

        self.ax_map.set_title(f"Real-Space Map [{mode}]", fontsize=11, fontweight='bold', pad=8, color='white')
        self.canvas_mpl.draw_idle()

    def update_stereo_projection_at_pixel(self, py: int, px: int):
        scany, scanx = self.state['scany'], self.state['scanx']
        if py < 0 or py >= scany or px < 0 or px >= scanx or self.R_matrices is None:
            return

        R = self.R_matrices[py, px]
        corr_val = float(self.corr_map[py, px]) if self.corr_map is not None else 0.0
        angle_val = float(self.angle_map[py, px]) if self.angle_map is not None else 0.0

        real_x = px * self.state['re_samp']
        real_y = (scany - 1 - py) * self.state['re_samp'] if self.state['left_origin'] == 'upper' else py * self.state['re_samp']
        self.cursor_marker.set_data([real_x], [real_y])

        if np.isnan(R[0, 0]) or corr_val < self.var_q_min.get():
            self.stereo_spot.set_data([], [])
            self.stereo_text.set_text(f"Pixel: ({py}, {px})\nUnindexed / Low Q\nQ = {corr_val:.2f}")
            self.canvas_mpl.draw_idle()
            return

        v_cart = R[:, 2]
        norm_v = np.linalg.norm(v_cart)
        if norm_v < 1e-6:
            return
        v_norm = v_cart / norm_v

        denom = 1.0 + abs(float(v_norm[2]))
        x_stereo = float(v_norm[0] / denom)
        y_stereo = float(v_norm[1] / denom)

        self.stereo_spot.set_data([x_stereo], [y_stereo])

        uvw_str = "[--]"
        if hasattr(self.state['acom_mgr'], 'extract_lattice_zone_axis'):
            uvw = self.state['acom_mgr'].extract_lattice_zone_axis({'matrix': R})
            uvw_str = f"[{uvw[0]} {uvw[1]} {uvw[2]}]"

        readout = (
            f"Pixel: ({py}, {px})\n"
            f"Zone Axis: {uvw_str}\n"
            f"In-Plane θ: {angle_val:+.1f}°\n"
            f"Quality Q:  {corr_val:.3f}\n"
            f"Stereo: ({x_stereo:+.2f}, {y_stereo:+.2f})"
        )
        self.stereo_text.set_text(readout)
        self.canvas_mpl.draw_idle()

    def _trigger_export(self):
        try:
            thresh_pct = 5.0
        except Exception:
            thresh_pct = 5.0
        q_min = float(self.var_q_min.get())
        self.export_callback(vacuum_masked=self.var_mask_vac.get(), thresh_pct=thresh_pct, q_min=q_min)


class SelectiveOrientationMapWindow(ctk.CTkToplevel):
    """
    Real-space grain and reflection reconstruction window for Bragg spots
    isolated within a designated reciprocal interval [q_min, q_max].
    """
    def __init__(self, parent, state, q_min: float, q_max: float, on_probe_select_cb=None, on_save_larmor_cb=None):
        super().__init__(parent)
        self.title(f"Selective In-Plane Orientation & Intensity Map [q ∈ {q_min:.3f} - {q_max:.3f} 1/nm]")
        self.geometry("1480x840")
        self.minsize(1100, 640)

        self.state = state
        self.q_min = float(min(q_min, q_max))
        self.q_max = float(max(q_min, q_max))
        self.on_probe_select_cb = on_probe_select_cb
        self.on_save_larmor_cb = on_save_larmor_cb

        self.scany = state['scany']
        self.scanx = state['scanx']
        self.rec_samp = float(state.get('rec_samp', 1.0))
        self.re_samp = float(state.get('re_samp', 1.0))

        self.M = np.array([
            [state.get('eps11', 1.0), state.get('eps12', 0.0)],
            [state.get('eps21', 0.0), state.get('eps22', 1.0)]
        ], dtype=np.float64)

        self.cyclic_colormaps = get_available_cyclic_colormaps()

        self.map_intensity = np.zeros((self.scany, self.scanx), dtype=np.float64)
        self.map_angle = np.full((self.scany, self.scanx), np.nan, dtype=np.float64)
        self.map_count = np.zeros((self.scany, self.scanx), dtype=np.int32)

        self._compute_filtered_maps()

        # Row 1: Primary Controls
        f_top1 = ctk.CTkFrame(self, height=42, corner_radius=0)
        f_top1.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(6, 2))

        ctk.CTkLabel(f_top1, text="View:", font=("Arial", 11, "bold")).pack(side=tk.LEFT, padx=(8, 2))
        self.var_view_mode = ctk.StringVar(value="Angle (Weighted RGB)")
        cb_mode = ctk.CTkOptionMenu(
            f_top1, variable=self.var_view_mode,
            values=["Angle (Weighted RGB)", "Dominant Angle θ (deg)", "Band Intensity", "Peak Count / Multiplicity"],
            width=175, command=lambda c: self.update_display()
        )
        cb_mode.pack(side=tk.LEFT, padx=3)

        ctk.CTkLabel(f_top1, text="n-Fold:", font=("Arial", 11, "bold"), text_color="#ff1744").pack(side=tk.LEFT, padx=(8, 2))
        self.var_multiplicity = ctk.StringVar(value="1")
        cb_mult = ctk.CTkOptionMenu(
            f_top1, variable=self.var_multiplicity,
            values=["1", "2", "3", "4", "5", "6", "7", "8"], width=65,
            command=lambda c: self.update_display()
        )
        cb_mult.pack(side=tk.LEFT, padx=3)

        ctk.CTkLabel(f_top1, text="Colormap:", font=("Arial", 11, "bold"), text_color="#00e5ff").pack(side=tk.LEFT, padx=(8, 2))
        first_cmap_label = list(self.cyclic_colormaps.keys())[0]
        self.var_cyclic_label = ctk.StringVar(value=first_cmap_label)
        cb_cmap = ctk.CTkOptionMenu(
            f_top1, variable=self.var_cyclic_label,
            values=list(self.cyclic_colormaps.keys()), width=180,
            command=lambda c: self.update_display()
        )
        cb_cmap.pack(side=tk.LEFT, padx=3)

        self.var_log_scale = tk.BooleanVar(value=False)
        chk_log = ctk.CTkSwitch(
            f_top1, text="Log Intensity", variable=self.var_log_scale,
            command=self.update_display, font=("Arial", 11, "bold"), width=110
        )
        chk_log.pack(side=tk.LEFT, padx=(8, 4))

        self.var_mask_vac = tk.BooleanVar(value=True)
        chk_mask = ctk.CTkCheckBox(
            f_top1, text="Mask Vac", variable=self.var_mask_vac,
            command=self.update_display, font=("Arial", 11)
        )
        chk_mask.pack(side=tk.LEFT, padx=(6, 4))

        ctk.CTkLabel(f_top1, text="Clip (%):", font=("Arial", 11)).pack(side=tk.LEFT, padx=(4, 2))
        self.var_clip = tk.DoubleVar(value=98.0)
        slider_clip = ctk.CTkSlider(
            f_top1, from_=80.0, to=100.0, number_of_steps=40,
            variable=self.var_clip, width=65, command=lambda v: self.update_display()
        )
        slider_clip.pack(side=tk.LEFT, padx=2)

        self.btn_export_pdf = ctk.CTkButton(
            f_top1, text="📄 PDF (300 DPI)", width=125,
            fg_color="#00838f", hover_color="#006064", command=self.export_pdf_300dpi
        )
        self.btn_export_pdf.pack(side=tk.RIGHT, padx=6)

        self.btn_export_npz = ctk.CTkButton(
            f_top1, text="💾 NPZ", width=50,
            fg_color="#37474f", hover_color="#263238", command=self.export_maps
        )
        self.btn_export_npz.pack(side=tk.RIGHT, padx=2)

        # Row 2: Secondary Controls (Larmor Offset & Wheel Controls)
        f_top2 = ctk.CTkFrame(self, height=38, corner_radius=0)
        f_top2.pack(side=tk.TOP, fill=tk.X, padx=8, pady=(0, 4))

        ctk.CTkLabel(f_top2, text="Larmor Offset (θ_off):", font=("Arial", 11, "bold"), text_color="#e040fb").pack(side=tk.LEFT, padx=(8, 2))
        
        init_larmor = float(self.state.get('larmor_offset_deg', 0.0))
        self.var_larmor_deg = tk.DoubleVar(value=init_larmor)
        self.lbl_larmor = ctk.CTkLabel(f_top2, text=f"{init_larmor:+.1f}°", font=("Courier", 11, "bold"), width=52)
        self.lbl_larmor.pack(side=tk.LEFT, padx=2)

        def _on_larmor_slider(v):
            val = float(v)
            self.lbl_larmor.configure(text=f"{val:+.1f}°")
            self.update_display()

        slider_larmor = ctk.CTkSlider(
            f_top2, from_=-180.0, to=180.0, number_of_steps=720,
            variable=self.var_larmor_deg, command=_on_larmor_slider, width=170
        )
        slider_larmor.set(init_larmor)
        slider_larmor.pack(side=tk.LEFT, padx=4)

        def _reset_larmor():
            slider_larmor.set(0.0)
            self.var_larmor_deg.set(0.0)
            self.lbl_larmor.configure(text="+0.0°")
            self.update_display()

        ctk.CTkButton(f_top2, text="↺ 0°", width=36, command=_reset_larmor).pack(side=tk.LEFT, padx=2)

        def _save_current_larmor():
            cur_val = float(self.var_larmor_deg.get())
            if self.on_save_larmor_cb:
                self.on_save_larmor_cb(cur_val)
                messagebox.showinfo(
                    "Preferences Saved",
                    f"Saved Larmor offset ({cur_val:+.1f}°) as default to:\n{CONFIG_FILE}",
                    parent=self
                )

        btn_save_larmor = ctk.CTkButton(
            f_top2, text="💾 Save Default", width=100,
            fg_color="#4a148c", hover_color="#311b92",
            command=_save_current_larmor
        )
        btn_save_larmor.pack(side=tk.LEFT, padx=(4, 10))

        ctk.CTkLabel(f_top2, text="Wheel Pos:", font=("Arial", 11, "bold"), text_color="#76ff03").pack(side=tk.LEFT, padx=(10, 2))
        self.var_wheel_pos = ctk.StringVar(value="Bottom-Right")
        cb_wheel_pos = ctk.CTkOptionMenu(
            f_top2, variable=self.var_wheel_pos,
            values=["Bottom-Right", "Top-Right", "Top-Left", "Bottom-Left"], width=110,
            command=lambda c: self.update_wheel_position()
        )
        cb_wheel_pos.pack(side=tk.LEFT, padx=3)

        ctk.CTkLabel(f_top2, text="Wheel Size:", font=("Arial", 11, "bold"), text_color="#ffca28").pack(side=tk.LEFT, padx=(8, 2))
        self.var_wheel_scale = tk.DoubleVar(value=0.14)
        self.lbl_wheel_scale = ctk.CTkLabel(f_top2, text="100%", font=("Courier", 11, "bold"), width=38)
        self.lbl_wheel_scale.pack(side=tk.LEFT, padx=2)

        def _on_wheel_scale_slider(v):
            val = float(v)
            pct = int(round((val / 0.14) * 100))
            self.lbl_wheel_scale.configure(text=f"{pct}%")
            self.update_wheel_position()

        slider_wheel_scale = ctk.CTkSlider(
            f_top2, from_=0.08, to=0.26, number_of_steps=90,
            variable=self.var_wheel_scale, command=_on_wheel_scale_slider, width=70
        )
        slider_wheel_scale.pack(side=tk.LEFT, padx=2)

        # Figure & Canvas Setup
        self.fig, self.ax = plt.subplots(figsize=(9.2, 6.5))
        self.fig.patch.set_facecolor('#1a1a1a')
        self.ax.set_facecolor('#121212')
        self.fig.subplots_adjust(top=0.92, bottom=0.08, left=0.08, right=0.90)

        # Polar Inset Axis
        init_box = self._get_wheel_bounding_box()
        self.ax_wheel = self.fig.add_axes(init_box, projection='polar')
        self.ax_wheel.set_facecolor('#1a1a1a')

        self.canvas_mpl = FigureCanvasTkAgg(self.fig, master=self)
        self.canvas_widget = self.canvas_mpl.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=4)

        self.im = None
        self.cbar = None
        self.cax = None
        self.cursor_marker = None

        self._init_plot()
        self.update_display()

        self.canvas_mpl.mpl_connect("button_press_event", self.on_click)

    def _compute_filtered_maps(self):
        """Filters peak coordinates per frame into refined q-interval reflections."""
        framepeaklist = self.state.get('framepeaklist', [])
        origin_upper = (self.state.get('right_origin', 'lower') == 'upper')

        for frame in framepeaklist:
            py = frame.get('i', None)
            px = frame.get('j', None)
            if py is None or px is None or py >= self.scany or px >= self.scanx:
                continue

            x_arr = frame.get('x', np.array([]))
            y_arr = frame.get('y', np.array([]))
            i_arr = frame.get('intensity', frame.get('raw_mass', frame.get('mass', np.array([]))))

            if len(x_arr) < 2:
                continue

            center_idx = int(np.argmax(i_arr))
            cx, cy = x_arr[center_idx], y_arr[center_idx]

            mask_diff = np.ones(len(x_arr), dtype=bool)
            mask_diff[center_idx] = False

            rel_x = x_arr[mask_diff] - cx
            rel_y = y_arr[mask_diff] - cy
            rel_i = i_arr[mask_diff]

            if len(rel_x) == 0:
                continue

            coords = np.column_stack([rel_x, rel_y])
            corr_coords = (self.M @ coords.T).T

            corr_qx = corr_coords[:, 0]
            corr_qy = -corr_coords[:, 1] if origin_upper else corr_coords[:, 1]

            q_vals = self.rec_samp * np.hypot(corr_qx, corr_qy)

            in_band = (q_vals >= self.q_min) & (q_vals <= self.q_max)
            if not np.any(in_band):
                continue

            band_qx = corr_qx[in_band]
            band_qy = corr_qy[in_band]
            band_i = rel_i[in_band]

            self.map_intensity[py, px] = float(np.sum(band_i))
            self.map_count[py, px] = int(np.sum(in_band))

            strongest_in_band = int(np.argmax(band_i))
            theta_deg = float(np.degrees(np.arctan2(band_qy[strongest_in_band], band_qx[strongest_in_band])))
            self.map_angle[py, px] = theta_deg

    def _get_active_cmap_name(self) -> str:
        label = self.var_cyclic_label.get()
        return self.cyclic_colormaps.get(label, "twilight")

    def _get_active_multiplicity(self) -> int:
        try:
            return max(1, min(8, int(self.var_multiplicity.get())))
        except (ValueError, TypeError):
            return 1

    def _get_wheel_bounding_box(self):
        size = float(self.var_wheel_scale.get())
        pos_key = self.var_wheel_pos.get()

        right_margin = 0.88 - size
        left_margin = 0.10
        top_margin = 0.88 - size
        bottom_margin = 0.11

        if pos_key == "Bottom-Right":
            return [right_margin, bottom_margin, size, size]
        elif pos_key == "Top-Right":
            return [right_margin, top_margin, size, size]
        elif pos_key == "Top-Left":
            return [left_margin, top_margin, size, size]
        else:
            return [left_margin, bottom_margin, size, size]

    def update_wheel_position(self):
        new_box = self._get_wheel_bounding_box()
        self.ax_wheel.set_position(new_box)
        self.canvas_mpl.draw_idle()

    def _draw_color_wheel(self, cmap_name: str, n_fold: int, larmor_deg: float):
        self.ax_wheel.clear()
        self.ax_wheel.set_facecolor('#1a1a1a')

        n_theta = 512
        n_r = 10
        theta = np.linspace(0.0, 2.0 * np.pi, n_theta)
        r = np.linspace(0.45, 1.0, n_r)
        T, R = np.meshgrid(theta, r)

        larmor_rad = np.radians(larmor_deg)
        delta_theta_rad = (2.0 * np.pi) / n_fold

        T_offset = T - larmor_rad
        phase_mesh = np.mod(T_offset, delta_theta_rad) / delta_theta_rad

        cmap_obj = plt.get_cmap(cmap_name).copy()
        self.ax_wheel.pcolormesh(T, R, phase_mesh, cmap=cmap_obj, shading='auto', zorder=2)

        self.ax_wheel.plot(theta, np.ones_like(theta) * 1.0, color='white', lw=0.8, zorder=3)
        self.ax_wheel.plot(theta, np.ones_like(theta) * 0.45, color='#37474f', lw=0.6, zorder=3)

        sector_deg = 360.0 / n_fold
        tick_degrees = [np.mod(k * sector_deg + larmor_deg, 360.0) for k in range(n_fold)]
        tick_radians = [np.radians(d) for d in tick_degrees]
        tick_labels = [f"{int(round(d))}°" for d in tick_degrees]

        self.ax_wheel.set_yticks([])
        self.ax_wheel.set_xticks(tick_radians)
        self.ax_wheel.set_xticklabels(tick_labels, color='white', fontsize=7.0, fontweight='bold')
        self.ax_wheel.tick_params(pad=1)
        self.ax_wheel.grid(color='#546e7a', linestyle=':', lw=0.7, alpha=0.7)
        self.ax_wheel.spines['polar'].set_visible(False)

        self.ax_wheel.text(
            0, 0, f"{n_fold}×", color='white', fontsize=8, ha='center', va='center',
            fontweight='bold', zorder=4
        )

    def _generate_weighted_rgb(self, cmap_name: str, n_fold: int, larmor_deg: float) -> np.ndarray:
        valid_int = self.map_intensity[self.map_intensity > 0]
        clip_pct = self.var_clip.get()
        max_int = float(np.percentile(valid_int, clip_pct)) if valid_int.size > 0 else 1.0

        theta_shifted = np.mod(self.map_angle - larmor_deg, 360.0)
        sector_deg = 360.0 / n_fold
        phase = np.nan_to_num(np.mod(theta_shifted, sector_deg) / sector_deg, nan=0.0)

        cmap = plt.get_cmap(cmap_name)
        base_rgb = cmap(phase)[..., :3]

        if self.var_log_scale.get():
            norm_intensity = np.log1p(np.maximum(0.0, self.map_intensity)) / np.log1p(max(1e-6, max_int))
            norm_intensity = np.clip(norm_intensity, 0.0, 1.0)
            weight = np.power(norm_intensity, 0.65)[..., np.newaxis]
        else:
            norm_intensity = np.clip(self.map_intensity / max(1e-6, max_int), 0.0, 1.0)
            weight = np.power(norm_intensity, 0.75)[..., np.newaxis]

        valid_angle_mask = (~np.isnan(self.map_angle))[..., np.newaxis]
        composite_rgb = base_rgb * weight * valid_angle_mask
        return np.clip(composite_rgb, 0.0, 1.0)

    def _init_plot(self):
        extent = [0, self.scanx * self.re_samp, 0, self.scany * self.re_samp]
        self.im = self.ax.imshow(
            np.zeros((self.scany, self.scanx)),
            extent=extent,
            origin=self.state.get('left_origin', 'lower'),
            interpolation='nearest'
        )
        self.ax.set_xlabel("x (nm)", fontsize=10, color='white')
        self.ax.set_ylabel("y (nm)", fontsize=10, color='white')
        self.ax.tick_params(colors='white', labelsize=9)

        div = make_axes_locatable(self.ax)
        self.cax = div.append_axes("right", size="3.5%", pad=0.08)
        self.cbar = self.fig.colorbar(self.im, cax=self.cax)
        self.cbar.ax.tick_params(colors='white', labelsize=8)

        self.cursor_marker = self.ax.plot([], [], 'r+', markersize=14, markeredgewidth=2.0, zorder=10)[0]

    def update_display(self):
        mode = self.var_view_mode.get()
        clip_pct = self.var_clip.get()
        cyclic_cmap = self._get_active_cmap_name()
        cyclic_label = self.var_cyclic_label.get()
        n_fold = self._get_active_multiplicity()
        larmor_deg = float(self.var_larmor_deg.get())
        sector_deg = 360.0 / n_fold

        mask = np.zeros((self.scany, self.scanx), dtype=bool)
        if self.var_mask_vac.get() and self.state.get('myset') is not None:
            if self.state.get('total_intensity_map') is None:
                self.state['total_intensity_map'] = np.sum(self.state['myset'].data, axis=(2, 3)).astype(np.float64)
            cutoff_val = 0.05 * np.max(self.state['total_intensity_map'])
            mask = self.state['total_intensity_map'] < cutoff_val

        if mode in ["Angle (Weighted RGB)", "Dominant Angle θ (deg)"]:
            self.ax_wheel.set_visible(True)
            self._draw_color_wheel(cyclic_cmap, n_fold, larmor_deg)
            self.update_wheel_position()
        else:
            self.ax_wheel.set_visible(False)

        if mode == "Angle (Weighted RGB)":
            rgb_disp = self._generate_weighted_rgb(cyclic_cmap, n_fold, larmor_deg)
            rgb_disp[mask] = 0.0
            self.im.set_data(rgb_disp)
            self.cax.set_visible(False)
            scale_type = "Log" if self.var_log_scale.get() else "Lin"
            self.ax.set_title(
                f"In-Plane θ ({cyclic_label}, {n_fold}-fold, θ_off={larmor_deg:+.1f}°) [{scale_type} Weighted]\n[q ∈ {self.q_min:.3f} - {self.q_max:.3f} 1/nm]",
                fontsize=11, fontweight='bold', color='white', pad=10
            )
        else:
            self.cax.set_visible(True)
            if mode == "Dominant Angle θ (deg)":
                theta_shifted = np.mod(self.map_angle - larmor_deg, 360.0)
                data = np.mod(theta_shifted, sector_deg)
                cmap_name = cyclic_cmap
                vmin, vmax = 0.0, sector_deg
                label = f"Angle θ (mod {sector_deg:.0f}°, θ_off={larmor_deg:+.1f}°)"
            elif mode == "Band Intensity":
                data = np.array(self.map_intensity, copy=True)
                cmap_name = "turbo"
                if self.var_log_scale.get():
                    data = np.log1p(data)
                    valid = data[data > 0]
                    label = "Log(1 + Band Intensity)"
                else:
                    valid = data[data > 0]
                    label = "Band Intensity (Linear)"
                vmin = 0.0
                vmax = float(np.percentile(valid, clip_pct)) if valid.size > 0 else 1.0
            else:
                data = np.array(self.map_count, dtype=np.float64, copy=True)
                cmap_name = "viridis"
                vmin = 0.0
                vmax = float(np.max(data)) if np.max(data) > 0 else 1.0
                label = "Bragg Spots in Band"

            data[mask] = np.nan
            cmap_obj = plt.get_cmap(cmap_name).copy()
            cmap_obj.set_bad(color='black', alpha=0.0)

            self.im.set_cmap(cmap_obj)
            self.im.set_clim(vmin, vmax)
            self.im.set_data(data)

            norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
            self.cbar.mappable.set_norm(norm)
            self.cbar.mappable.set_clim(vmin, vmax)
            self.cbar.set_label(label, fontsize=9, color='white')
            self.cbar.update_normal(self.im)
            self.ax.set_title(
                f"Selective Map: {mode} ({n_fold}-fold) [q ∈ {self.q_min:.3f} - {self.q_max:.3f} 1/nm]",
                fontsize=11, fontweight='bold', color='white', pad=10
            )

        self.canvas_mpl.draw_idle()

    def on_click(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None or event.button != 1:
            return

        px = int(np.clip(event.xdata / self.re_samp, 0, self.scanx - 1))
        if self.state.get('left_origin', 'lower') == 'upper':
            py = int(np.clip(self.scany - 1 - (event.ydata / self.re_samp), 0, self.scany - 1))
        else:
            py = int(np.clip(event.ydata / self.re_samp, 0, self.scany - 1))

        real_x = px * self.re_samp
        real_y = (self.scany - 1 - py) * self.re_samp if self.state.get('left_origin', 'lower') == 'upper' else py * self.re_samp
        self.cursor_marker.set_data([real_x], [real_y])
        self.canvas_mpl.draw_idle()

        if self.on_probe_select_cb is not None:
            self.on_probe_select_cb(py, px)

    def export_pdf_300dpi(self):
        default_name = f"{self.state.get('filebasename', 'Map')}_q_{self.q_min:.2f}_{self.q_max:.2f}_{self.var_multiplicity.get()}fold.pdf"
        fn = filedialog.asksaveasfilename(
            parent=self,
            initialfile=default_name,
            defaultextension=".pdf",
            filetypes=[
                ("Portable Document Format (*.pdf)", "*.pdf"),
                ("High-Res PNG (*.png)", "*.png"),
                ("TIFF Image (*.tif)", "*.tif"),
                ("All Files", "*.*")
            ]
        )
        if not fn:
            return

        cursor_was_vis = self.cursor_marker.get_visible()
        self.cursor_marker.set_visible(False)
        try:
            self.fig.savefig(
                fn,
                dpi=300,
                facecolor=self.fig.get_facecolor(),
                edgecolor='none',
                bbox_inches='tight',
                pad_inches=0.08
            )
            messagebox.showinfo(
                "Export Complete",
                f"Saved publication map at 300 DPI:\n{os.path.basename(fn)}",
                parent=self
            )
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to save figure: {e}", parent=self)
        finally:
            self.cursor_marker.set_visible(cursor_was_vis)
            self.canvas_mpl.draw_idle()

    def export_maps(self):
        fn = filedialog.asksaveasfilename(
            parent=self,
            defaultextension=".npz",
            filetypes=[("NumPy Compressed", "*.npz"), ("All Files", "*.*")]
        )
        if fn:
            try:
                n_fold = self._get_active_multiplicity()
                larmor_deg = float(self.var_larmor_deg.get())
                np.savez_compressed(
                    fn,
                    intensity=self.map_intensity.astype(np.float32),
                    angle=self.map_angle.astype(np.float32),
                    count=self.map_count.astype(np.int32),
                    composite_rgb=self._generate_weighted_rgb(self._get_active_cmap_name(), n_fold, larmor_deg),
                    multiplicity=np.int32(n_fold),
                    larmor_offset_deg=np.float32(larmor_deg),
                    log_scaled=self.var_log_scale.get(),
                    q_interval=np.array([self.q_min, self.q_max], dtype=np.float32)
                )
                messagebox.showinfo("Export Complete", f"Saved selective maps to:\n{os.path.basename(fn)}", parent=self)
            except Exception as e:
                messagebox.showerror("Export Error", str(e), parent=self)
        
class LatticeStrainDialog(ctk.CTkToplevel):
    def __init__(self, parent, state, on_recalculate_callback):
        super().__init__(parent)
        self.title("Lattice Strain Analysis")
        self.geometry("480x520")
        self.resizable(False, False)

        self.state = state
        self.on_recalculate_callback = on_recalculate_callback

        f_main = ctk.CTkFrame(self, corner_radius=10)
        f_main.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        ctk.CTkLabel(f_main, text="Lattice Strain & Deformation Tensor", font=("Arial", 13, "bold"), text_color="#00e5ff").pack(anchor='w', padx=10, pady=(6, 2))

        self.lbl_subtitle = ctk.CTkLabel(f_main, text="Matched Spots: 0 | Zone Axis: [u v w]", font=("Arial", 11, "italic"))
        self.lbl_subtitle.pack(anchor='w', padx=10, pady=(0, 6))

        f_tol = ctk.CTkFrame(f_main, fg_color="transparent")
        f_tol.pack(fill=tk.X, padx=10, pady=(2, 6))

        self.lbl_tol = ctk.CTkLabel(f_tol, text=f"Match Tolerance: {self.state.get('strain_tolerance', 0.35):.2f} 1/nm", font=("Arial", 11, "bold"), width=190, anchor='w')
        self.lbl_tol.pack(side=tk.LEFT, padx=(0, 4))

        self.slider_tol = ctk.CTkSlider(f_tol, from_=0.05, to=1.50, number_of_steps=145, width=210, command=self._on_slider_move)
        self.slider_tol.set(self.state.get('strain_tolerance', 0.35))
        self.slider_tol.pack(side=tk.LEFT, padx=4)

        self.f_data = ctk.CTkFrame(f_main)
        self.f_data.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        self.value_labels = {}
        row_keys = [
            ("eps_xx", "ε_xx (Real-Space Normal):"),
            ("eps_yy", "ε_yy (Real-Space Normal):"),
            ("eps_xy", "ε_xy (Engineering Shear):"),
            ("omega", "Rigid Rotation (ω):"),
            ("eps_1", "Principal Strain ε_1:"),
            ("eps_2", "Principal Strain ε_2:"),
            ("theta_p", "Principal Axis Angle:"),
            ("mean_res", "Mean Residual Error:")
        ]

        for i, (k, label_txt) in enumerate(row_keys):
            row = ctk.CTkFrame(self.f_data, fg_color="transparent")
            row.pack(fill=tk.X, padx=10, pady=3)
            ctk.CTkLabel(row, text=label_txt, anchor='w', font=("Arial", 11)).pack(side=tk.LEFT)
            val_lbl = ctk.CTkLabel(row, text="--", anchor='e', font=("Courier", 11, "bold"))
            val_lbl.pack(side=tk.RIGHT)
            self.value_labels[k] = val_lbl

        f_btn = ctk.CTkFrame(f_main, fg_color="transparent")
        f_btn.pack(fill=tk.X, padx=10, pady=(10, 4))
        ctk.CTkButton(f_btn, text="Close", width=80, command=self.destroy).pack(side=tk.RIGHT, padx=4)

    def _on_slider_move(self, val):
        r_val = float(val)
        self.state['strain_tolerance'] = r_val
        self.lbl_tol.configure(text=f"Match Tolerance: {r_val:.2f} 1/nm")
        if self.on_recalculate_callback:
            self.on_recalculate_callback(self)

    def update_results(self, strain_dict):
        n_pairs = strain_dict.get('matched_pairs', 0)
        zone_str = str(strain_dict.get('zone_axis', '[u v w]'))
        self.lbl_subtitle.configure(text=f"Matched Spots: {n_pairs} | Zone Axis: {zone_str}")

        eps_xx = strain_dict.get('eps_xx', 0.0) * 100.0
        eps_yy = strain_dict.get('eps_yy', 0.0) * 100.0
        eps_xy = strain_dict.get('eps_xy', 0.0) * 100.0
        omega_deg = np.degrees(strain_dict.get('omega', 0.0))
        eps_1 = strain_dict.get('eps_1', 0.0) * 100.0
        eps_2 = strain_dict.get('eps_2', 0.0) * 100.0
        theta_p = np.degrees(strain_dict.get('theta_p', 0.0))
        mean_res = strain_dict.get('mean_residual', 0.0)

        self.value_labels['eps_xx'].configure(text=f"{eps_xx:+.3f} %")
        self.value_labels['eps_yy'].configure(text=f"{eps_yy:+.3f} %")
        self.value_labels['eps_xy'].configure(text=f"{eps_xy:+.3f} %")
        self.value_labels['omega'].configure(text=f"{omega_deg:+.3f}°")
        self.value_labels['eps_1'].configure(text=f"{eps_1:+.3f} %")
        self.value_labels['eps_2'].configure(text=f"{eps_2:+.3f} %")
        self.value_labels['theta_p'].configure(text=f"{theta_p:+.2f}°")
        self.value_labels['mean_res'].configure(text=f"{mean_res:.4f} 1/nm")


class TemplateLibraryDialog(ctk.CTkToplevel):
    def __init__(self, parent, acom_mgr, on_library_changed=None):
        super().__init__(parent)
        self.title("Kinematical Orientation Library Manager (py4DSTEM)")
        self.geometry("540x450")
        self.resizable(False, False)
        self.grab_set()

        self.mgr = acom_mgr
        self.on_library_changed = on_library_changed

        f = ctk.CTkFrame(self, corner_radius=10)
        f.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        ctk.CTkLabel(f, text="Precalculated Kinematical Orientation Plan", font=("Arial", 13, "bold")).pack(anchor='w', padx=10, pady=(6, 8))

        row_cif = ctk.CTkFrame(f, fg_color="transparent")
        row_cif.pack(fill=tk.X, padx=10, pady=4)
        ctk.CTkLabel(row_cif, text="CIF File:").pack(side=tk.LEFT, padx=4)
        self.var_cif = ctk.StringVar(value=self.mgr.cif_path)
        ctk.CTkEntry(row_cif, textvariable=self.var_cif, width=320).pack(side=tk.LEFT, padx=4)
        ctk.CTkButton(row_cif, text="Browse", width=60, command=self.browse_cif).pack(side=tk.LEFT)

        row_par = ctk.CTkFrame(f, fg_color="transparent")
        row_par.pack(fill=tk.X, padx=10, pady=4)
        ctk.CTkLabel(row_par, text="kV:").pack(side=tk.LEFT, padx=2)
        self.var_kv = ctk.StringVar(value="200")
        ctk.CTkEntry(row_par, textvariable=self.var_kv, width=42).pack(side=tk.LEFT, padx=2)

        ctk.CTkLabel(row_par, text="k_max:").pack(side=tk.LEFT, padx=(6, 2))
        self.var_kmax = ctk.StringVar(value="2.0")
        ctk.CTkEntry(row_par, textvariable=self.var_kmax, width=42).pack(side=tk.LEFT, padx=2)

        ctk.CTkLabel(row_par, text="Step (°):").pack(side=tk.LEFT, padx=(6, 2))
        self.var_step = ctk.StringVar(value="1.0")
        ctk.CTkEntry(row_par, textvariable=self.var_step, width=42).pack(side=tk.LEFT, padx=2)

        ctk.CTkLabel(row_par, text="σ_s (1/Å):").pack(side=tk.LEFT, padx=(6, 2))
        self.var_sigma_s = ctk.StringVar(value=str(getattr(self.mgr, 'sigma_excitation_error', 0.02)))
        ctk.CTkEntry(row_par, textvariable=self.var_sigma_s, width=45).pack(side=tk.LEFT, padx=2)

        self.btn_gen = ctk.CTkButton(f, text="⚡ Generate Orientation Plan", fg_color="#00897b", hover_color="#00695c", command=self.generate_plan)
        self.btn_gen.pack(fill=tk.X, padx=10, pady=(10, 6))

        ctk.CTkFrame(f, height=2, fg_color="#37474f").pack(fill=tk.X, padx=10, pady=8)

        row_io = ctk.CTkFrame(f, fg_color="transparent")
        row_io.pack(fill=tk.X, padx=10, pady=4)
        ctk.CTkButton(row_io, text="💾 Save Database", command=self.save_db).pack(side=tk.LEFT, padx=6, expand=True)
        ctk.CTkButton(row_io, text="📁 Load Database", command=self.load_db).pack(side=tk.LEFT, padx=6, expand=True)

        self.lbl_status = ctk.CTkLabel(f, text="Status: Ready", font=("Arial", 10, "italic"))
        self.lbl_status.pack(anchor='w', padx=12, pady=(12, 4))

        ctk.CTkButton(f, text="Done", width=80, fg_color="transparent", border_width=1, command=self.destroy).pack(anchor='e', padx=10, pady=6)

    def browse_cif(self):
        fn = filedialog.askopenfilename(
            parent=self,
            filetypes=[("CIF Crystal Files", "*.cif"), ("All Files", "*.*")]
        )
        if fn:
            self.var_cif.set(fn)

    def generate_plan(self):
        cif = self.var_cif.get().strip()
        if not os.path.exists(cif):
            messagebox.showerror("Error", "Please provide a valid CIF file path.", parent=self)
            return

        try:
            self.lbl_status.configure(text="Generating symmetry-reduced orientation library... Check console.")
            self.update_idletasks()

            sigma_s = float(self.var_sigma_s.get())
            res = self.mgr.build_library_from_cif(
                cif,
                accel_voltage=float(self.var_kv.get()) * 1e3,
                k_max=float(self.var_kmax.get()),
                angular_step_deg=float(self.var_step.get()),
                sigma_excitation_error=sigma_s
            )
            self.lbl_status.configure(text=res)
            if self.on_library_changed:
                self.on_library_changed()
            messagebox.showinfo("Success", res, parent=self)
        except Exception as e:
            self.lbl_status.configure(text=f"Failed: {e}")
            messagebox.showerror("Simulation Error", str(e), parent=self)

    def save_db(self):
        fn = filedialog.asksaveasfilename(
            parent=self,
            defaultextension=".pkl",
            filetypes=[
                ("Pickle DB", "*.pkl"),
                ("Gzip Compressed DB", "*.gz"),
                ("Zipped Archive DB", "*.zip"),
                ("All Files", "*.*")
            ]
        )
        if fn:
            try:
                self.mgr.save_library(fn)
                self.lbl_status.configure(text=f"Saved: {os.path.basename(fn)}")
            except Exception as e:
                messagebox.showerror("Save Error", str(e), parent=self)

    def load_db(self):
        fn = filedialog.askopenfilename(
            parent=self,
            filetypes=[
                ("Pickle DB", "*.pkl"),
                ("Gzip Compressed DB", "*.gz"),
                ("Zipped Archive DB", "*.zip"),
                ("Compressed Pickle", "*.pklz"),
                ("All Files", "*.*")
            ]
        )
        if not fn:
            return

        base_name = os.path.basename(fn)
        self.lbl_status.configure(text=f"Loading: {base_name} ...")
        self.update_idletasks()

        try:
            self.mgr.load_library(fn)
            self.var_sigma_s.set(str(getattr(self.mgr, 'sigma_excitation_error', 0.02)))
            self.lbl_status.configure(text=f"Loaded: {base_name}")
            if self.on_library_changed:
                self.on_library_changed()
            messagebox.showinfo("Loaded", f"Loaded library for:\n{self.mgr.cif_path}", parent=self)
        except Exception as e:
            self.lbl_status.configure(text=f"Failed to load: {e}")
            messagebox.showerror("Load Error", str(e), parent=self)


class PeakDetectionConfigDialog(ctk.CTkToplevel):
    def __init__(self, parent, state, app_config, on_apply_callback):
        super().__init__(parent)
        self.title("Peak Detection Parameters")
        self.geometry("500x740")
        self.resizable(False, False)

        self.state = state
        self.app_config = app_config
        self.on_apply_callback = on_apply_callback

        f_main = ctk.CTkFrame(self, corner_radius=10)
        f_main.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        preset_name = self.state.get('detector_preset', 'ELA')
        f_title = ctk.CTkFrame(f_main, fg_color="transparent")
        f_title.pack(fill=tk.X, padx=10, pady=(6, 4))
        
        self.lbl_title = ctk.CTkLabel(
            f_title, 
            text=f"Peak Detection Settings ({preset_name})", 
            font=("Arial", 13, "bold")
        )
        self.lbl_title.pack(side=tk.LEFT)

        f_fields = ctk.CTkScrollableFrame(f_main, height=540)
        f_fields.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

        params = self.state['peak_params']
        self.vars = {}

        ctk.CTkLabel(f_fields, text="Core Detection", font=("Arial", 11, "bold"), text_color="#00e5ff").grid(
            row=0, column=0, columnspan=2, sticky='w', padx=6, pady=(4, 2)
        )

        lbl_scale = ctk.CTkLabel(f_fields, text="Fitting Intensity Scale:", anchor='w')
        lbl_scale.grid(row=1, column=0, sticky='w', padx=6, pady=4)

        is_lin = params.get('fit_scale_linear', False)
        self.var_scale_mode = ctk.StringVar(value="Linear" if is_lin else "Log")
        self.seg_scale = ctk.CTkSegmentedButton(f_fields, values=["Linear", "Log"], variable=self.var_scale_mode, width=130)
        self.seg_scale.grid(row=1, column=1, sticky='e', padx=6, pady=4)

        init_pct = float(params.get('peak_keep_percent', 100.0))
        self.var_keep_pct = tk.DoubleVar(value=init_pct)

        self.lbl_keep = ctk.CTkLabel(f_fields, text=f"Keep Strongest Peaks: {int(init_pct)}%", anchor='w')
        self.lbl_keep.grid(row=2, column=0, sticky='w', padx=6, pady=(6, 2))

        def _on_pct_change(val):
            pct_val = int(round(float(val)))
            self.lbl_keep.configure(text=f"Keep Strongest Peaks: {pct_val}%")
            self.state['peak_params']['peak_keep_percent'] = float(pct_val)

        slider_keep = ctk.CTkSlider(
            f_fields,
            from_=5.0,
            to=100.0,
            number_of_steps=95,
            variable=self.var_keep_pct,
            command=_on_pct_change,
            width=160
        )
        slider_keep.grid(row=2, column=1, sticky='e', padx=6, pady=(6, 2))

        primary_field_defs = [
            ("feature_size", "Feature Diameter (odd px):", int, params.get('feature_size', 7)),
            ("feature_threshold", "Feature Threshold:", float, params.get('feature_threshold', 0.5)),
            ("feature_minmass", "Min Integrated Mass:", float, params.get('feature_minmass', 2.0))
        ]

        curr_row = 3
        for key, label_txt, val_type, current_val in primary_field_defs:
            lbl = ctk.CTkLabel(f_fields, text=label_txt, anchor='w')
            lbl.grid(row=curr_row, column=0, sticky='w', padx=6, pady=4)

            var = ctk.StringVar(value=str(current_val))
            widget = ctk.CTkEntry(f_fields, textvariable=var, width=120)
            widget.grid(row=curr_row, column=1, sticky='e', padx=6, pady=4)

            self.vars[key] = (var, val_type)
            curr_row += 1

        f_divider = ctk.CTkFrame(f_fields, height=2, fg_color="#37474f")
        f_divider.grid(row=curr_row, column=0, columnspan=2, sticky="ew", padx=6, pady=(12, 6))
        curr_row += 1

        lbl_sep = ctk.CTkLabel(f_fields, text="Fine-Tuning & Filtering Controls", font=("Arial", 11, "bold"), text_color="#90a4ae")
        lbl_sep.grid(row=curr_row, column=0, columnspan=2, sticky='w', padx=6, pady=(2, 6))
        curr_row += 1

        secondary_field_defs = [
            ("feature_separation", "Min Separation (px):", int, params.get('feature_separation', 8)),
            ("edge_margin", "Edge Margin (px):", int, params.get('edge_margin', 5)),
            ("exclude_center_radius", "Direct Beam Exclude (1/nm):", float, params.get('exclude_center_radius', 0.5)),
            ("feature_percentile", "Threshold Percentile (%):", float, params.get('feature_percentile', 0.0)),
            ("noise_size", "Noise Size (px, std 1.0):", float, params.get('noise_size', 1.0)),
            ("smooth_size", "Boxcar Smooth Size (0=None):", int, params.get('smooth_size', 0) or 0),
            ("frame_cutoff", "Frame Cutoff Floor:", float, params.get('frame_cutoff', 0.0)),
            ("frame_cmax", "Contrast Clip Max:", float, params.get('frame_cmax', 10.0)),
            ("conv2D", "Perform 2D Convolution:", bool, params.get('conv2D', False))
        ]

        for key, label_txt, val_type, current_val in secondary_field_defs:
            lbl = ctk.CTkLabel(f_fields, text=label_txt, anchor='w')
            lbl.grid(row=curr_row, column=0, sticky='w', padx=6, pady=4)

            if val_type == bool:
                var = tk.BooleanVar(value=bool(current_val))
                widget = ctk.CTkCheckBox(f_fields, text="", variable=var, width=28)
            else:
                var = ctk.StringVar(value=str(current_val))
                widget = ctk.CTkEntry(f_fields, textvariable=var, width=120)

            widget.grid(row=curr_row, column=1, sticky='e', padx=6, pady=4)
            self.vars[key] = (var, val_type)
            curr_row += 1

        f_btn = ctk.CTkFrame(f_main, fg_color="transparent")
        f_btn.pack(fill=tk.X, padx=10, pady=(10, 4))

        ctk.CTkButton(f_btn, text="Apply", width=80, command=self.apply_params).pack(side=tk.LEFT, padx=3)
        ctk.CTkButton(
            f_btn, 
            text="⚡ Apply & Fit", 
            width=110, 
            fg_color="#00897b", 
            hover_color="#00695c", 
            command=self.apply_and_fit
        ).pack(side=tk.LEFT, padx=3)
        ctk.CTkButton(f_btn, text="↺ Defaults", width=85, fg_color="#546e7a", hover_color="#37474f", command=self.reset_to_defaults).pack(side=tk.LEFT, padx=3)
        ctk.CTkButton(f_btn, text="Close", width=70, fg_color="transparent", border_width=1, command=self.destroy).pack(side=tk.RIGHT, padx=3)

    def apply_params(self):
        is_lin = (self.var_scale_mode.get() == "Linear")
        self.state['peak_params']['fit_scale_linear'] = is_lin
        self.state['peak_params']['peak_keep_percent'] = float(self.var_keep_pct.get())
        self.app_config['fit_scale_linear'] = is_lin
        save_config(self.app_config)

        for key, (var, val_type) in self.vars.items():
            val_str = str(var.get()).strip()
            if val_type == bool:
                self.state['peak_params'][key] = bool(var.get())
            elif val_type == int:
                try:
                    self.state['peak_params'][key] = int(float(val_str))
                except ValueError:
                    pass
            elif val_type == float:
                try:
                    self.state['peak_params'][key] = float(val_str)
                except ValueError:
                    pass

    def apply_and_fit(self):
        self.apply_params()
        if self.on_apply_callback:
            self.on_apply_callback()

    def reset_to_defaults(self):
        for k, v in DEFAULT_PEAK_PARAMS.items():
            self.state['peak_params'][k] = v
            if k in self.vars:
                var, val_type = self.vars[k]
                if val_type == bool:
                    var.set(bool(v))
                else:
                    var.set(str(v if v is not None else 0))
        self.var_scale_mode.set("Linear" if DEFAULT_PEAK_PARAMS['fit_scale_linear'] else "Log")
        self.var_keep_pct.set(DEFAULT_PEAK_PARAMS.get('peak_keep_percent', 100.0))
        self.lbl_keep.configure(text=f"Keep Strongest Peaks: {int(DEFAULT_PEAK_PARAMS.get('peak_keep_percent', 100.0))}%")


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
        self.geometry("520x780")  # Slightly enlarged to accommodate the new row
        self.resizable(False, False)
        self.grab_set()

        self.cfg = current_cfg.copy()
        self.updated = False

        f_main = ctk.CTkFrame(self, corner_radius=10)
        f_main.pack(fill=tk.BOTH, expand=True, padx=15, pady=15)

        # Theme
        f_theme = ctk.CTkFrame(f_main)
        f_theme.pack(fill=tk.X, padx=10, pady=5)
        ctk.CTkLabel(f_theme, text="GUI Theme:", font=("Arial", 12, "bold")).pack(side=tk.LEFT, padx=10)
        self.var_theme = ctk.StringVar(value=self.cfg.get("theme", "dark"))
        cb_theme = ctk.CTkOptionMenu(f_theme, variable=self.var_theme, values=["dark", "light"], width=100)
        cb_theme.pack(side=tk.RIGHT, padx=10, pady=5)

        # Database Path
        f_db = ctk.CTkFrame(f_main)
        f_db.pack(fill=tk.X, padx=10, pady=5)
        ctk.CTkLabel(f_db, text="Calibration DB Path:", font=("Arial", 12, "bold")).pack(anchor='w', padx=10, pady=(6,0))
        
        f_db_inner = ctk.CTkFrame(f_db, fg_color="transparent")
        f_db_inner.pack(fill=tk.X, padx=10, pady=5)
        self.var_db_path = ctk.StringVar(value=self.cfg.get("cal_db_path", ""))
        ctk.CTkEntry(f_db_inner, textvariable=self.var_db_path, width=320).pack(side=tk.LEFT, padx=(0, 6))
        ctk.CTkButton(f_db_inner, text="Browse...", width=80, command=self.browse_db).pack(side=tk.LEFT)

        # Display Preferences
        f_disp = ctk.CTkFrame(f_main)
        f_disp.pack(fill=tk.X, padx=10, pady=5)
        #cmap_opts = ['gray', 'gray_r', 'inferno', 'magma', 'viridis', 'plasma', 'turbo', 'cividis', 'hsv', 'twilight', 'coolwarm']
        cmap_opts = get_available_display_colormaps()
        origin_opts = ['lower', 'upper']

        row1 = ctk.CTkFrame(f_disp, fg_color="transparent")
        row1.pack(fill=tk.X, padx=10, pady=3)
        ctk.CTkLabel(row1, text="Left Cmap:").pack(side=tk.LEFT, padx=4)
        self.var_l_cmap = ctk.StringVar(value=self.cfg.get("left_cmap", "gray"))
        ctk.CTkOptionMenu(row1, variable=self.var_l_cmap, values=cmap_opts, width=110).pack(side=tk.LEFT, padx=4)

        ctk.CTkLabel(row1, text="Right Cmap:").pack(side=tk.LEFT, padx=(12, 4))
        self.var_r_cmap = ctk.StringVar(value=self.cfg.get("right_cmap", "turbo"))
        ctk.CTkOptionMenu(row1, variable=self.var_r_cmap, values=cmap_opts, width=110).pack(side=tk.LEFT, padx=4)

        row_ori = ctk.CTkFrame(f_disp, fg_color="transparent")
        row_ori.pack(fill=tk.X, padx=10, pady=3)
        ctk.CTkLabel(row_ori, text="Left Origin:").pack(side=tk.LEFT, padx=4)
        self.var_l_origin = ctk.StringVar(value=self.cfg.get("left_origin", "lower"))
        ctk.CTkOptionMenu(row_ori, variable=self.var_l_origin, values=origin_opts, width=110).pack(side=tk.LEFT, padx=4)

        ctk.CTkLabel(row_ori, text="Right Origin:").pack(side=tk.LEFT, padx=(12, 4))
        self.var_r_origin = ctk.StringVar(value=self.cfg.get("right_origin", "lower"))
        ctk.CTkOptionMenu(row_ori, variable=self.var_r_origin, values=origin_opts, width=110).pack(side=tk.LEFT, padx=4)

        row2 = ctk.CTkFrame(f_disp, fg_color="transparent")
        row2.pack(fill=tk.X, padx=10, pady=3)
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

        # --- Larmor Rotation Offset Section ---
        f_larmor = ctk.CTkFrame(f_main)
        f_larmor.pack(fill=tk.X, padx=10, pady=5)
        ctk.CTkLabel(f_larmor, text="Larmor Rotation Offset (Scan vs Camera):", font=("Arial", 11, "bold")).pack(anchor='w', padx=10, pady=(4, 2))

        row_larmor = ctk.CTkFrame(f_larmor, fg_color="transparent")
        row_larmor.pack(fill=tk.X, padx=10, pady=(2, 6))

        init_larmor = float(self.cfg.get("larmor_offset_deg", 0.0))
        self.var_larmor = ctk.StringVar(value=f"{init_larmor:+.1f}")
        self.var_larmor_slider = tk.DoubleVar(value=init_larmor)

        def _on_slider(v):
            val = float(v)
            self.var_larmor.set(f"{val:+.1f}")

        def _on_entry(*args):
            try:
                val = float(self.var_larmor.get())
                val = np.clip(val, -180.0, 180.0)
                self.var_larmor_slider.set(val)
            except ValueError:
                pass

        self.var_larmor.trace_add("write", _on_entry)

        self.slider_larmor = ctk.CTkSlider(
            row_larmor, from_=-180.0, to=180.0, number_of_steps=720,
            variable=self.var_larmor_slider, command=_on_slider, width=220
        )
        self.slider_larmor.pack(side=tk.LEFT, padx=(0, 8), fill=tk.X, expand=True)

        entry_larmor = ctk.CTkEntry(row_larmor, textvariable=self.var_larmor, width=65, font=("Courier", 11, "bold"))
        entry_larmor.pack(side=tk.LEFT, padx=4)
        ctk.CTkLabel(row_larmor, text="deg").pack(side=tk.LEFT, padx=2)

        def _reset_larmor_val():
            self.slider_larmor.set(0.0)
            self.var_larmor.set("+0.0")

        ctk.CTkButton(row_larmor, text="↺ 0°", width=42, command=_reset_larmor_val).pack(side=tk.LEFT, padx=4)

        # Custom Ring Presets
        f_presets = ctk.CTkFrame(f_main)
        f_presets.pack(fill=tk.X, padx=10, pady=5)
        ctk.CTkLabel(f_presets, text="Custom Ring Presets (JSON Dictionary q in 1/nm):", font=("Arial", 11, "bold")).pack(anchor='w', padx=10, pady=(4, 2))

        self.txt_presets = ctk.CTkTextbox(f_presets, height=90, font=("Courier", 11))
        self.txt_presets.pack(fill=tk.X, padx=10, pady=4)
        presets_dict = self.cfg.get("ring_presets", DEFAULT_RING_PRESETS)
        self.txt_presets.insert("1.0", json.dumps(presets_dict, indent=2))

        # Distortion Matrix M
        f_dist = ctk.CTkFrame(f_main)
        f_dist.pack(fill=tk.X, padx=10, pady=5)
        ctk.CTkLabel(f_dist, text="Reciprocal Distortion Matrix M:", font=("Arial", 11, "bold")).pack(anchor='w', padx=10, pady=(4, 2))

        m_grid = ctk.CTkFrame(f_dist, fg_color="transparent")
        m_grid.pack(padx=10, pady=2)

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

        # Buttons
        f_btn = ctk.CTkFrame(f_main, fg_color="transparent")
        f_btn.pack(pady=8)
        ctk.CTkButton(f_btn, text="Save & Apply", command=self.on_save, width=110).pack(side=tk.LEFT, padx=6)
        ctk.CTkButton(f_btn, text="Cancel", command=self.destroy, fg_color="transparent", border_width=1, width=90).pack(side=tk.LEFT, padx=6)

    def browse_db(self):
        fn = filedialog.askopenfilename(title="Select Calibration JSON", filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")])
        if fn: self.var_db_path.set(fn)

    def on_save(self):
        try:
            self.cfg["ring_presets"] = json.loads(self.txt_presets.get("1.0", tk.END).strip())
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
            self.cfg["larmor_offset_deg"] = float(self.var_larmor.get())  # <-- Saved
            self.cfg["eps11"] = float(self.var_eps11.get())
            self.cfg["eps12"] = float(self.var_eps12.get())
            self.cfg["eps21"] = float(self.var_eps21.get())
            self.cfg["eps22"] = float(self.var_eps22.get())
            save_config(self.cfg)
            self.updated = True
            self.destroy()
        except Exception as e:
            messagebox.showerror("Validation Error", str(e), parent=self)
            

class FileImportDialog(ctk.CTkToplevel):
    def __init__(self, parent, initial_cfg):
        super().__init__(parent)
        self.title("4D-STEM Dataset Import")
        self.geometry("500x380")
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
            f_fmt, variable=self.var_format, 
            values=["Auto-detect", "DECTRIS", "plain DECTRIS", "EMPAD", "PantaRhei"], 
            width=150, command=self._on_format_change
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
        ctk.CTkLabel(f_bin, text="Binning Settings", font=("Arial", 11, "bold")).pack(anchor='w', padx=10, pady=(6, 2))

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
        if fn_lower.endswith('.raw'): return 'EMPAD'
        elif fn_lower.endswith(('.prz', '.prz.zip')) or 'pantarhei' in fn_lower: return 'PantaRhei'
        elif fn_lower.endswith(('.h5', '.hdf5')): return 'DECTRIS'
        return 'DECTRIS'

    def browse_file(self):
        filename = filedialog.askopenfilename(
            title="Select 4D-STEM Dataset",
            filetypes=[("All Supported", "*.h5 *.hdf5 *.raw *.prz *.zip *.npz"), ("All Files", "*.*")]
        )
        if filename:
            self.var_filepath.set(filename)
            fmt = self.auto_detect_format(filename)
            self.var_format.set(fmt)
            self._on_format_change(fmt)

    def on_ok(self):
        filepath = self.var_filepath.get().strip()
        if not filepath or not os.path.exists(filepath):
            messagebox.showerror("Invalid File", "Please select a valid dataset file.", parent=self)
            return
        try:
            bs_y, bs_x = int(self.var_bs_y.get()), int(self.var_bs_x.get())
            bd_y, bd_x = int(self.var_bd_y.get()), int(self.var_bd_x.get())
        except ValueError:
            messagebox.showerror("Invalid Binning", "Binning must be integers.", parent=self)
            return

        chosen_fmt = self.var_format.get()
        if chosen_fmt == "Auto-detect":
            chosen_fmt = self.auto_detect_format(filepath)

        scan_shape = None
        if chosen_fmt == "plain DECTRIS":
            try:
                scan_shape = (int(self.var_scany.get()), int(self.var_scanx.get()))
            except ValueError:
                messagebox.showerror("Error", "Invalid scan shape.", parent=self)
                return

        filedir, filename = os.path.split(filepath)
        basename, suffix = os.path.splitext(filename)
        actual_type = "DECTRIS" if chosen_fmt in ["DECTRIS", "plain DECTRIS"] else chosen_fmt

        self.result = {
            "full_path": filepath,
            "path": filedir + "/",
            "filebasename": basename,
            "filesuffix": suffix,
            "format_type": actual_type,
            "scan_shape": scan_shape,
            "is_zipfile": filepath.lower().endswith(('.zip', '.prz.zip')),
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

        ctk.CTkLabel(f_main, text="Override pixel calibrations:", font=("Arial", 12, "bold")).pack(anchor='w', padx=10, pady=(6, 10))

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
        ctk.CTkButton(f_btn, text="Apply", command=self.on_apply, width=120).pack(side=tk.LEFT, padx=6)
        ctk.CTkButton(f_btn, text="Cancel", command=self.destroy, fg_color="transparent", border_width=1, width=80).pack(side=tk.LEFT, padx=6)

    def on_apply(self):
        try:
            self.result = {"re_samp": float(self.var_re_samp.get()), "rec_samp": float(self.var_rec_samp.get())}
            self.destroy()
        except ValueError as e:
            messagebox.showerror("Error", str(e), parent=self)


class DistortionCalibDialog(ctk.CTkToplevel):
    def __init__(self, parent, presets_dict, current_M, update_calib_rings_cb, close_calib_cb, app_config):
        super().__init__(parent)
        self.title("Distortion Matrix Calibration")
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
        cb_preset = ctk.CTkOptionMenu(f_preset, variable=self.var_preset, values=preset_names, command=lambda c: self.notify_update())
        cb_preset.pack(side=tk.RIGHT, padx=10, pady=6)

        f_sliders = ctk.CTkFrame(f_main)
        f_sliders.pack(fill=tk.X, padx=10, pady=6)

        self.sliders, self.labels, self.vars = {}, {}, {}
        keys = [('eps11', 0, 0), ('eps12', 0, 1), ('eps21', 1, 0), ('eps22', 1, 1)]
        
        for key, r, c in keys:
            base_val = self.base_M[r, c]
            delta = abs(base_val) * 0.06 if abs(base_val) > 1e-5 else 0.06
            min_v, max_v = base_val - delta, base_val + delta

            f_s = ctk.CTkFrame(f_sliders, fg_color="transparent")
            f_s.pack(fill=tk.X, padx=8, pady=4)

            lbl = ctk.CTkLabel(f_s, text=f"ε{r+1}{c+1}: {base_val:.4f}", width=90, anchor='w')
            lbl.pack(side=tk.LEFT, padx=4)
            self.labels[key] = lbl

            var_s = tk.DoubleVar(value=base_val)
            self.vars[key] = var_s

            slider = ctk.CTkSlider(f_s, from_=min_v, to=max_v, variable=var_s, command=lambda val, k=key, row=r, col_idx=c: self.on_slider_move(k, row, col_idx, val))
            slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)
            self.sliders[key] = slider

            ctk.CTkButton(f_s, text="↺", width=28, command=lambda k=key, row=r, col_idx=c, bv=base_val: self.reset_slider(k, row, col_idx, bv)).pack(side=tk.RIGHT, padx=2)

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
        self.notify_update()

    def _format_matrix_str(self):
        return f"[ {self.curr_M[0,0]:.5f} ,  {self.curr_M[0,1]:.5f} ]\n[ {self.curr_M[1,0]:.5f} ,  {self.curr_M[1,1]:.5f} ]"

    def on_slider_move(self, key, row, col, value):
        val_float = float(value)
        self.curr_M[row, col] = val_float
        self.labels[key].configure(text=f"ε{row+1}{col+1}: {val_float:.4f}")
        self.lbl_matrix.configure(text=self._format_matrix_str())
        self.notify_update()

    def reset_slider(self, key, row, col, base_val):
        self.vars[key].set(base_val)
        self.on_slider_move(key, row, col, base_val)

    def notify_update(self):
        radii_q = self.presets.get(self.var_preset.get(), [])
        self.update_calib_rings_cb(radii_q, self.curr_M)

    def transfer_to_settings(self):
        det = np.linalg.det(self.curr_M)
        norm_M = self.curr_M / np.sqrt(abs(det)) if abs(det) > 1e-9 else self.curr_M
        self.app_config['eps11'] = float(norm_M[0, 0])
        self.app_config['eps12'] = float(norm_M[0, 1])
        self.app_config['eps21'] = float(norm_M[1, 0])
        self.app_config['eps22'] = float(norm_M[1, 1])
        save_config(self.app_config)
        self.transferred = True
        messagebox.showinfo("Transferred", f"Transferred to Settings:\n{self._format_matrix_str()}", parent=self)

    def on_close(self):
        self.close_calib_cb()
        self.destroy()


# ==============================================================================
# Main STEM Explorer Function
# ==============================================================================
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
    init_preset = config.get("detector_preset", "ELA")

    if myset is not None and hasattr(myset, 'PreparePeakDetectionPars'):
        init_peak_params = myset.PreparePeakDetectionPars(shortcut=init_preset)
    else:
        init_peak_params = DEFAULT_PEAK_PARAMS.copy()

    init_peak_params['fit_scale_linear'] = config.get("fit_scale_linear", False)

    acom_instance = ACOMManager()
    acom_instance.sigma_excitation_error = config.get("sigma_excitation_error", 0.02)

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
        'calib_active': False,
        'detector_preset': init_preset,
        'peak_params': init_peak_params,
        'peaks': [],
        'norm_img': None,
        'show_stats': False,
        'acom_mgr': acom_instance,
        'orientation_candidates': [],
        'selected_candidate_idx': 0,
        'template_rot_offset': 0.0,
        'show_template_overlay': True,
        'template_intensity_thresh': 0.01,
        'strain_tolerance': 0.35,
        'strain_dialog_ref': None,
        'use_map_roi': False,
        'map_roi': {'xmin': 0, 'xmax': 0, 'ymin': 0, 'ymax': 0},
        'strain_maps': {
            'eps_xx': None,
            'eps_yy': None,
            'eps_xy': None,
            'omega': None
        },
        'orientation_maps': {
            'angle': None,
            'corr': None,
            'zone_axis_id': None,
            'R_matrices': None
        },
        'total_intensity_map': None,
        'framepeaklist': [],
        'larmor_offset_deg': float(config.get("larmor_offset_deg", 0.0))
    }

    pan_state = {'active': False, 'ax': None, 'start_x': 0, 'start_y': 0, 'xlim': None, 'ylim': None}
    click_tracker = {'last_time': 0, 'last_ax': None}
    ruler1_state = {'active': False, 'p1': None, 'p2': None, 'length': 0.0}
    ruler2_state = {'active': False, 'p1': None, 'p2': None, 'dq': 0.0, 'd_spacing': 0.0}

    mapping_win_ref = [None]
    orient_win_ref = [None]
    peak_dialog_ref = [None] 
    powder_win_ref = [None]
    selective_win_ref = [None]
    divider_cache = [None]

    artists = {
        'ring_patches': [], 
        'calib_patches': [], 
        'peak_markers': [], 
        'template_patches': [],
        'strain_lines': [],
        'map_roi_rect': None,
        'cbar1': None
    }

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
        existing_files = glob.glob(f"{save_prefix}_[0-9][0-9][0-9].{save_fmt}")
        max_idx = 0
        for fname in existing_files:
            match = re.search(r'_(\d{3})\.' + re.escape(save_fmt) + r'$', fname)
            if match and int(match.group(1)) > max_idx:
                max_idx = int(match.group(1))
        next_idx = max_idx + 1
        return next_idx, f"{save_prefix}_{next_idx:03d}.{save_fmt}"

    def sync_target_dict():
        target_dict['x'] = int(state['x'])
        target_dict['y'] = int(state['y'])
        target_dict['re_samp'] = float(state['re_samp'])
        target_dict['rec_samp'] = float(state['rec_samp'])
        target_dict['dx'] = int(state['dx'])
        target_dict['dy'] = int(state['dy'])
        target_dict['cqx'] = float(state['cqx'])
        target_dict['cqy'] = float(state['cqy'])
        target_dict['r_in'] = float(state['r_in'])
        target_dict['r_out'] = float(state['r_out'])
        target_dict['mode'] = state['calc_mode']
        target_dict['peaks'] = list(state['peaks'])
        target_dict['strain_tolerance'] = float(state['strain_tolerance'])

    def redraw_ring_markers():
        for p in artists['ring_patches']:
            try: p.remove()
            except Exception: pass
        artists['ring_patches'].clear()

        if state['myset'] is None or state['calib_active']: return
        for q_radius in state['ring_markers']:
            circ = Circle((0, 0), q_radius, color='cyan', fill=False, linestyle=':', linewidth=1.2, alpha=0.85, zorder=5)
            ax2.add_patch(circ)
            artists['ring_patches'].append(circ)

    def draw_calib_elliptical_rings(radii_q, test_M):
        for p in artists['calib_patches']:
            try: p.remove()
            except Exception: pass
        artists['calib_patches'].clear()
        if state['myset'] is None: return

        try: inv_M = np.linalg.inv(test_M)
        except np.linalg.LinAlgError: inv_M = np.eye(2)

        theta = np.linspace(0, 2*np.pi, 120)
        for q_r in radii_q:
            poly_xy = (inv_M @ np.vstack([q_r * np.cos(theta), q_r * np.sin(theta)])).T
            poly = Polygon(poly_xy, closed=True, edgecolor='#ff9800', facecolor='none', linestyle='--', linewidth=1.4, zorder=5)
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
        if preset_key in state['ring_presets']:
            q_radii = state['ring_presets'][preset_key]
            state['ring_markers'] = list(q_radii)
            if q_radii and 'var_ring_q' in globals_or_locals():
                var_ring_q.set(str(q_radii[0]))
            sync_target_dict()
            redraw_ring_markers()
            canvas.draw_idle()

    def globals_or_locals():
        return locals()

    def redraw_peak_markers():
        for p in artists['peak_markers']:
            try: p.remove()
            except Exception: pass
        artists['peak_markers'].clear()

        if not state['peaks']: return
        marker_rad = max(0.04, 4.0 * state['rec_samp'])
        for pt in state['peaks']:
            circ = Circle((pt['qx'], pt['qy']), radius=marker_rad, edgecolor='#ff1744', facecolor='none', linewidth=1.5, alpha=0.9, zorder=6)
            ax2.add_patch(circ)
            artists['peak_markers'].append(circ)

    def redraw_template_spots():
        for p in artists['template_patches']:
            try: p.remove()
            except Exception: pass
        artists['template_patches'].clear()

        if not state.get('show_template_overlay', True):
            canvas.draw_idle()
            return

        candidates = state.get('orientation_candidates', [])
        if not candidates:
            canvas.draw_idle()
            return

        idx = state.get('selected_candidate_idx', 0)
        if idx >= len(candidates): return
        cand = candidates[idx]

        d_theta = np.radians(state.get('template_rot_offset', 0.0))
        cos_t, sin_t = np.cos(d_theta), np.sin(d_theta)

        spots_data = cand.get('sim_spots', [])
        if not spots_data: return

        intensities = np.array([pt[3] if len(pt) >= 4 else 1.0 for pt in spots_data], dtype=np.float64)
        max_I = np.max(intensities) if np.max(intensities) > 0 else 1.0
        rel_I = intensities / max_I

        cutoff = float(state.get('template_intensity_thresh', 0.01))

        for pt, norm_I in zip(spots_data, rel_I):
            if norm_I < cutoff:
                continue

            qx_base, qy_base, hkl = pt[0], pt[1], pt[2]
            qx_rot = cos_t * qx_base - sin_t * qy_base
            qy_rot = sin_t * qx_base + cos_t * qy_base

            weight = norm_I ** 0.35
            msize = 4.0 + 18.0 * weight
            alpha_val = float(np.clip(0.18 + 0.62 * weight, 0.15, 0.85))

            circ = ax2.plot(
                qx_rot, qy_rot, 
                marker='o', 
                markersize=msize, 
                markerfacecolor='#76ff03', 
                markeredgecolor='#ffffff', 
                markeredgewidth=1.1, 
                alpha=alpha_val, 
                zorder=7
            )[0]
            artists['template_patches'].append(circ)
            
            if (abs(qx_rot) > 0.35 or abs(qy_rot) > 0.35) and norm_I >= max(0.05, cutoff):
                hkl_str = f"{hkl[0]}{hkl[1]}{hkl[2]}"
                txt = ax2.text(
                    qx_rot + 0.06, qy_rot + 0.06, 
                    hkl_str, 
                    color='#76ff03', 
                    fontsize=7, 
                    alpha=min(1.0, alpha_val + 0.25), 
                    zorder=8
                )
                artists['template_patches'].append(txt)

        canvas.draw_idle()

    def update_image_statistics():
        if 'stats_box_right' in artists:
            if not state['show_stats'] or state['norm_img'] is None:
                artists['stats_box_right'].set_visible(False)
            else:
                img_r = state['norm_img']
                d_min, d_max = float(np.min(img_r)), float(np.max(img_r))
                d_mean, d_std = float(np.mean(img_r)), float(np.std(img_r))
                d_sum = float(np.sum(img_r))
                p50, p90, p99 = np.percentile(img_r, [50, 90, 99])

                txt_r = (
                    f"Recip Frame Stats:\n"
                    f"Sum:  {d_sum:.3e}\n"
                    f"Mean: {d_mean:.3e}  Std: {d_std:.3e}\n"
                    f"Min:  {d_min:.3e}  Max: {d_max:.3e}\n"
                    f"p50:  {p50:.3e}  p90: {p90:.3e}  p99: {p99:.3e}"
                )
                artists['stats_box_right'].set_text(txt_r)
                artists['stats_box_right'].set_visible(True)

        if 'stats_box_left' in artists:
            if not state['show_stats'] or 'data_left_raw' not in state or state['data_left_raw'] is None:
                artists['stats_box_left'].set_visible(False)
            else:
                img_l = state['data_left_raw']
                valid_vals = img_l[~np.isnan(img_l)]
                if valid_vals.size > 0:
                    l_min, l_max = float(np.min(valid_vals)), float(np.max(valid_vals))
                    l_mean, l_std = float(np.mean(valid_vals)), float(np.std(valid_vals))
                    l_sum = float(np.sum(valid_vals))
                    p50, p90, p99 = np.percentile(valid_vals, [50, 90, 99])
                    txt_l = (
                        f"Real Space Stats [{state['calc_mode']}]:\n"
                        f"Sum:  {l_sum:.3e}\n"
                        f"Mean: {l_mean:.3e}  Std: {l_std:.3e}\n"
                        f"Min:  {l_min:.3e}  Max: {l_max:.3e}\n"
                        f"p50:  {p50:.3e}  p90: {p90:.3e}  p99: {p99:.3e}"
                    )
                else:
                    txt_l = "Real Space Stats:\nNo valid data"
                artists['stats_box_left'].set_text(txt_l)
                artists['stats_box_left'].set_visible(True)

    def update_map_roi_visual():
        if artists.get('map_roi_rect') is not None:
            try:
                artists['map_roi_rect'].remove()
            except Exception:
                pass
            artists['map_roi_rect'] = None

        if not state.get('use_map_roi', False) or state['myset'] is None:
            return

        roi = state.get('map_roi', {})
        xmin, xmax = roi.get('xmin', 0), roi.get('xmax', 0)
        ymin, ymax = roi.get('ymin', 0), roi.get('ymax', 0)

        width_nm = (xmax - xmin + 1) * state['re_samp']
        height_nm = (ymax - ymin + 1) * state['re_samp']

        x_left_nm = xmin * state['re_samp']
        if state['left_origin'] == 'upper':
            y_bottom_nm = (state['scany'] - 1 - ymax) * state['re_samp']
        else:
            y_bottom_nm = ymin * state['re_samp']

        rect = Rectangle(
            (x_left_nm, y_bottom_nm), width_nm, height_nm,
            linewidth=1.8, edgecolor='#00e5ff', facecolor='none', linestyle='-.', zorder=5
        )
        ax1.add_patch(rect)
        artists['map_roi_rect'] = rect

    def align_diffraction_centers():
        if state['myset'] is None:
            messagebox.showwarning("No Dataset", "Please load a dataset before aligning diffraction centers.", parent=root)
            return

        data_4d = state['myset'].data
        sy, sx, dy, dx = data_4d.shape
        flat_data = data_4d.reshape(sy, sx, -1)
        cy_map, cx_map = np.unravel_index(np.argmax(flat_data, axis=-1), (dy, dx))
        y_coords, x_coords = np.mgrid[:sy, :sx]
        A = np.column_stack([np.ones(sy * sx), x_coords.ravel(), y_coords.ravel()])
        coeff_y, _, _, _ = np.linalg.lstsq(A, cy_map.ravel(), rcond=None)
        coeff_x, _, _, _ = np.linalg.lstsq(A, cx_map.ravel(), rcond=None)
        fitted_cy = (A @ coeff_y).reshape(sy, sx)
        fitted_cx = (A @ coeff_x).reshape(sy, sx)

        target_cy = int(round(np.mean(fitted_cy)))
        target_cx = int(round(np.mean(fitted_cx)))
        shift_y = np.round(target_cy - fitted_cy).astype(int)
        shift_x = np.round(target_cx - fitted_cx).astype(int)

        for y_idx in range(sy):
            for x_idx in range(sx):
                dy_s, dx_s = shift_y[y_idx, x_idx], shift_x[y_idx, x_idx]
                if dy_s != 0 or dx_s != 0:
                    data_4d[y_idx, x_idx, :, :] = np.roll(data_4d[y_idx, x_idx, :, :], shift=(dy_s, dx_s), axis=(0, 1))

        state['cqy'], state['cqx'] = target_cy, target_cx
        rebuild_plot()
        messagebox.showinfo("Complete", f"Frames aligned to center: ({target_cy}, {target_cx}) px.", parent=root)

    def run_peak_detection():
        if state['myset'] is None:
            messagebox.showwarning("No Dataset", "Please load a dataset before running peak detection.")
            return
        try:
            if state['norm_img'] is None: refresh_spatial_roi()
            norm_img = state['norm_img'].copy()
            if norm_img is None: return

            use_lin = state['peak_params'].get('fit_scale_linear', False)
            peaklist = PeakDetection(norm_img, params=state['peak_params'], origin=state['right_origin'], lin=use_lin)

            state['peaks'].clear()
            geom_cx, geom_cy = state['detx'] / 2.0, state['dety'] / 2.0

            if peaklist:
                res_dict = peaklist[0] if isinstance(peaklist, list) else peaklist
                px_rel_x   = np.atleast_1d(res_dict.get('x', []))
                px_rel_y   = np.atleast_1d(res_dict.get('y', []))
                masses     = np.atleast_1d(res_dict.get('mass', np.ones_like(px_rel_x)))
                
                if 'raw_mass' in res_dict and len(res_dict['raw_mass']) == len(px_rel_x):
                    raw_masses = np.atleast_1d(res_dict['raw_mass'])
                else:
                    raw_masses = masses.copy()

                det_px_x = px_rel_x + geom_cx
                det_px_y = px_rel_y + geom_cy

                dists_to_curr_center = np.hypot(det_px_x - state['cqx'], det_px_y - state['cqy'])
                near_center_mask = dists_to_curr_center < 25.0
                
                if np.any(near_center_mask):
                    near_indices = np.where(near_center_mask)[0]
                    central_idx = near_indices[np.argmax(raw_masses[near_indices])]
                    
                    true_center_x = float(det_px_x[central_idx])
                    true_center_y = float(det_px_y[central_idx])

                    state['cqx'] = true_center_x
                    state['cqy'] = true_center_y
                else:
                    true_center_x = state['cqx']
                    true_center_y = state['cqy']

                feat_sz = int(state['peak_params'].get('feature_size', 7))
                if feat_sz % 2 == 0:
                    feat_sz += 1
                spot_area = max(1.0, (np.pi / 4.0) * (feat_sz ** 2))

                if not use_lin:
                    mean_log_val = np.maximum(0.0, raw_masses / spot_area)
                    lin_masses = np.expm1(mean_log_val) * spot_area
                else:
                    lin_masses = raw_masses.copy()

                edge_margin = max(0, int(state['peak_params'].get('edge_margin', 5)))
                r_center_excl = float(state['peak_params'].get('exclude_center_radius', 0.5))
                det_dim_x = state['detx']
                det_dim_y = state['dety']

                for x_px, y_px, m, rm, lm in zip(det_px_x, det_px_y, masses, raw_masses, lin_masses):
                    if (x_px < edge_margin or x_px >= (det_dim_x - edge_margin) or
                        y_px < edge_margin or y_px >= (det_dim_y - edge_margin)):
                        continue

                    qx = (x_px - true_center_x) * state['rec_samp']
                    if state['right_origin'] == 'upper':
                        qy = (true_center_y - y_px) * state['rec_samp']
                    else:
                        qy = (y_px - true_center_y) * state['rec_samp']

                    if np.hypot(qx, qy) < r_center_excl:
                        continue

                    state['peaks'].append({
                        'qx': float(qx),
                        'qy': float(qy),
                        'mass': float(m),
                        'raw_mass': float(rm),
                        'intensity': float(lm),
                        'det_x': float(x_px),
                        'det_y': float(y_px)
                    })

                keep_pct = float(state['peak_params'].get('peak_keep_percent', 100.0))
                if keep_pct < 100.0 and len(state['peaks']) > 3:
                    state['peaks'].sort(key=lambda p: p.get('intensity', p['mass']), reverse=True)
                    n_keep = max(3, int(np.ceil(len(state['peaks']) * (keep_pct / 100.0))))
                    state['peaks'] = state['peaks'][:n_keep]

            refresh_detector_mask()
            redraw_peak_markers()
            sync_target_dict()
            canvas.draw_idle()
            lbl_status.configure(text=f"Detected {len(state['peaks'])} peaks (Top {int(state['peak_params'].get('peak_keep_percent', 100))}%, Margin: {int(state['peak_params'].get('edge_margin', 5))}px).")
        except Exception as e:
            messagebox.showerror("Peak Detection Error", str(e))
            
    def clear_peak_detection():
        state['peaks'].clear()
        for p in artists['strain_lines']:
            try: p.remove()
            except Exception: pass
        artists['strain_lines'].clear()
        redraw_peak_markers()
        sync_target_dict()
        canvas.draw_idle()
        lbl_status.configure(text="Diffraction peaks cleared.")

    def compute_lattice_strain(dialog_instance=None):
        candidates = state.get('orientation_candidates', [])
        peaks = state.get('peaks', [])

        if not candidates:
            messagebox.showwarning("No Orientation", "Please index an orientation plan before calculating strain.", parent=root)
            return
        if len(peaks) < 3:
            messagebox.showwarning("Insufficient Peaks", "At least 3 experimental peaks required.", parent=root)
            return

        cand = candidates[state.get('selected_candidate_idx', 0)]
        d_theta = np.radians(state.get('template_rot_offset', 0.0))
        cos_t, sin_t = np.cos(d_theta), np.sin(d_theta)

        sim_coords = []
        for spot in cand['sim_spots']:
            qx_b, qy_b = spot[0], spot[1]
            qx_r = cos_t * qx_b - sin_t * qy_b
            qy_r = sin_t * qx_b + cos_t * qy_b
            sim_coords.append([qx_r, qy_r])
        sim_coords = np.array(sim_coords)

        r_cutoff = float(state.get('strain_tolerance', 0.35))
        matched_exp, matched_sim = [], []

        for p in peaks:
            p_exp = np.array([p['qx'], p['qy']])
            dists = np.linalg.norm(sim_coords - p_exp, axis=1)
            min_idx = int(np.argmin(dists))
            if dists[min_idx] <= r_cutoff:
                matched_exp.append(p_exp)
                matched_sim.append(sim_coords[min_idx])

        matched_exp = np.array(matched_exp)
        matched_sim = np.array(matched_sim)

        for p in artists['strain_lines']:
            try: p.remove()
            except Exception: pass
        artists['strain_lines'].clear()

        if len(matched_exp) < 3:
            if dialog_instance is None:
                messagebox.showerror("Match Failure", f"Only {len(matched_exp)} spot pairs matched within {r_cutoff:.2f} 1/nm.", parent=root)
            canvas.draw_idle()
            return

        X, Y = matched_sim, matched_exp
        A, _, _, _ = np.linalg.lstsq(X, Y, rcond=None)
        A = A.T

        try:
            T = np.linalg.inv(A).T
        except np.linalg.LinAlgError:
            if dialog_instance is None:
                messagebox.showerror("Math Error", "Singular transformation matrix.", parent=root)
            return

        eps_xx = float(T[0, 0] - 1.0)
        eps_yy = float(T[1, 1] - 1.0)
        eps_xy = float(0.5 * (T[0, 1] + T[1, 0]))
        omega = float(0.5 * (T[0, 1] - T[1, 0]))

        center = 0.5 * (eps_xx + eps_yy)
        R = np.hypot(0.5 * (eps_xx - eps_yy), eps_xy)
        eps_1 = float(center + R)
        eps_2 = float(center - R)
        theta_p = float(0.5 * np.arctan2(2.0 * eps_xy, eps_xx - eps_yy))

        residuals = np.linalg.norm(Y - (X @ A.T), axis=1)
        mean_res = float(np.mean(residuals))

        for (px, py), (sx, sy) in zip(matched_exp, matched_sim):
            line = ax2.plot([sx, px], [sy, py], color='#00e5ff', lw=1.2, linestyle=':', zorder=9)[0]
            artists['strain_lines'].append(line)
        canvas.draw_idle()

        strain_dict = {
            'matched_pairs': len(matched_exp),
            'zone_axis': cand['zone_axis'],
            'eps_xx': eps_xx,
            'eps_yy': eps_yy,
            'eps_xy': eps_xy,
            'omega': omega,
            'eps_1': eps_1,
            'eps_2': eps_2,
            'theta_p': theta_p,
            'mean_residual': mean_res
        }

        if dialog_instance is not None and dialog_instance.winfo_exists():
            dialog_instance.update_results(strain_dict)
        else:
            if state['strain_dialog_ref'] is not None and state['strain_dialog_ref'].winfo_exists():
                state['strain_dialog_ref'].lift()
                state['strain_dialog_ref'].update_results(strain_dict)
            else:
                dlg = LatticeStrainDialog(root, state, on_recalculate_callback=compute_lattice_strain)
                dlg.update_results(strain_dict)
                state['strain_dialog_ref'] = dlg

    def open_peak_params_dialog():
        if peak_dialog_ref[0] is not None and peak_dialog_ref[0].winfo_exists():
            peak_dialog_ref[0].lift()
            return
        peak_dialog_ref[0] = PeakDetectionConfigDialog(root, state, config, on_apply_callback=run_peak_detection)

    def handle_strain_export(vacuum_masked: bool, thresh_pct: float):
        if state['strain_maps']['eps_xx'] is None:
            messagebox.showwarning("No Data", "Run 'Map 2D Strain' first.", parent=root)
            return

        default_fn = f"{state['filebasename']}_strain_maps.npz"
        fn = filedialog.asksaveasfilename(
            title="Export 2D Strain Maps", initialdir=state['path'], initialfile=default_fn,
            filetypes=[("NumPy Compressed", "*.npz"), ("32-bit TIFF Stack", "*.tif *.tiff"), ("All Supported", "*.npz *.tif *.tiff")]
        )
        if fn:
            try:
                valid_mask = None
                if vacuum_masked:
                    if state['total_intensity_map'] is None:
                        state['total_intensity_map'] = np.sum(state['myset'].data, axis=(2, 3)).astype(np.float64)
                    valid_mask = state['total_intensity_map'] >= ((thresh_pct / 100.0) * np.max(state['total_intensity_map']))

                meta = {
                    'dataset': f"{state['filebasename']}{state['filesuffix']}",
                    're_samp_nm': state['re_samp'],
                    'rec_samp_q': state['rec_samp'],
                    'scan_dim': (state['scany'], state['scanx']),
                    'strain_tolerance': state.get('strain_tolerance', 0.35),
                    'in_plane_offset_deg': state.get('template_rot_offset', 0.0),
                    'vacuum_mask_applied': vacuum_masked,
                    'vacuum_mask_threshold_pct': thresh_pct,
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                }
                res = export_strain_maps_to_file(fn, state['strain_maps'], valid_mask=valid_mask, metadata=meta)
                lbl_status.configure(text=res)
                messagebox.showinfo("Export Complete", f"Exported strain maps to:\n{fn}", parent=root)
            except Exception as e:
                messagebox.showerror("Export Error", str(e), parent=root)

    def open_mapping_window():
        if state['strain_maps']['eps_xx'] is None:
            messagebox.showwarning("No Maps", "Please run 'Map 2D Strain' first.", parent=root)
            return
        if mapping_win_ref[0] is not None and mapping_win_ref[0].winfo_exists():
            mapping_win_ref[0].lift()
            mapping_win_ref[0].update_maps()
        else:
            mapping_win_ref[0] = StrainMappingWindow(root, state, export_callback=handle_strain_export)

    def handle_orientation_export(vacuum_masked: bool, thresh_pct: float, q_min: float = 0.0):
        orient_maps = state.get('orientation_maps', {})
        if orient_maps.get('R_matrices') is None:
            messagebox.showwarning("No Data", "Run 'Map Orientation' first.", parent=root)
            return

        default_fn = f"{state['filebasename']}_orientation_maps.npz"
        fn = filedialog.asksaveasfilename(
            title="Export 2D Orientation Maps", initialdir=state['path'], initialfile=default_fn,
            filetypes=[("NumPy Compressed", "*.npz"), ("32-bit TIFF Stack", "*.tif *.tiff"), ("All Supported", "*.npz *.tif *.tiff")]
        )
        if fn:
            try:
                R_matrices = np.array(orient_maps['R_matrices'], copy=True)
                corr_grid = np.array(orient_maps['corr'], copy=True)
                angle_grid = np.array(orient_maps['angle'], copy=True)

                invalid_mask = corr_grid < q_min
                if vacuum_masked:
                    if state.get('total_intensity_map') is None:
                        state['total_intensity_map'] = np.sum(state['myset'].data, axis=(2, 3)).astype(np.float64)
                    cutoff_val = (thresh_pct / 100.0) * np.max(state['total_intensity_map'])
                    invalid_mask |= (state['total_intensity_map'] < cutoff_val)

                R_matrices[invalid_mask] = np.nan
                corr_grid[invalid_mask] = 0.0
                angle_grid[invalid_mask] = np.nan

                meta = {
                    'dataset': f"{state['filebasename']}{state['filesuffix']}",
                    're_samp_nm': state['re_samp'],
                    'rec_samp_q': state['rec_samp'],
                    'scan_dim': (state['scany'], state['scanx']),
                    'cif_path': getattr(state['acom_mgr'], 'cif_path', ''),
                    'q_min_threshold': q_min,
                    'vacuum_masked': vacuum_masked,
                    'vacuum_cutoff_pct': thresh_pct,
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                }
                
                payload = {
                    'R_matrices': R_matrices.astype(np.float32),
                    'corr': corr_grid.astype(np.float32),
                    'in_plane_angle': angle_grid.astype(np.float32),
                    'metadata': json.dumps(meta, indent=2, default=str)
                }

                if fn.lower().endswith('.npz'):
                    np.savez_compressed(fn, **payload)
                elif fn.lower().endswith(('.tif', '.tiff')):
                    flat_R = [R_matrices[:, :, r, c] for r in range(3) for c in range(3)]
                    stack = np.stack(flat_R + [corr_grid, angle_grid], axis=0).astype(np.float32)
                    if HAS_TIFFFILE:
                        tifffile.imwrite(fn, stack, photometric='minisblack')
                    else:
                        from PIL import Image
                        Image.fromarray(corr_grid.astype(np.float32)).save(fn)

                lbl_status.configure(text=f"Exported orientation maps to: {os.path.basename(fn)}")
                messagebox.showinfo("Export Complete", f"Exported orientation maps to:\n{fn}", parent=root)
            except Exception as e:
                messagebox.showerror("Export Error", str(e), parent=root)

    def set_probe_position_from_map(py: int, px: int):
        if state['myset'] is None:
            return
        
        scany, scanx = state['scany'], state['scanx']
        state['x'] = int(np.clip(px, 0, scanx - 1))
        state['y'] = int(np.clip(py, 0, scany - 1))

        real_x = state['x'] * state['re_samp']
        if state['left_origin'] == 'upper':
            real_y = (scany - 1 - state['y']) * state['re_samp']
        else:
            real_y = state['y'] * state['re_samp']

        artists['v_line'].set_xdata([real_x, real_x])
        artists['h_line'].set_ydata([real_y, real_y])

        refresh_spatial_roi()
        canvas.draw_idle()

    def open_orient_mapping_window():
        orient_maps = state.get('orientation_maps', {})
        if orient_maps.get('R_matrices') is None:
            messagebox.showwarning("No Maps", "Please run 'Map Orientation' first.", parent=root)
            return
        if orient_win_ref[0] is not None and orient_win_ref[0].winfo_exists():
            orient_win_ref[0].lift()
            orient_win_ref[0].update_spatial_map()
        else:
            orient_win_ref[0] = OrientationMappingWindow(
                root, 
                state, 
                export_callback=handle_orientation_export,
                on_probe_select_callback=set_probe_position_from_map
            )

    def on_powder_range_selected(q_min, q_max):
        """Callback receiving isolated q-bounds from PowderPlotDialog to open SelectiveOrientationMapWindow."""
        lbl_status.configure(text=f"Selected powder range: q ∈ [{q_min:.3f}, {q_max:.3f}] 1/nm")

        if selective_win_ref[0] is not None and selective_win_ref[0].winfo_exists():
            selective_win_ref[0].destroy()

        def _save_larmor_to_config(larmor_val):
            config['larmor_offset_deg'] = float(larmor_val)
            state['larmor_offset_deg'] = float(larmor_val)
            save_config(config)

        selective_win_ref[0] = SelectiveOrientationMapWindow(
            root,
            state,
            q_min=q_min,
            q_max=q_max,
            on_probe_select_cb=set_probe_position_from_map,
            on_save_larmor_cb=_save_larmor_to_config
        )

    def open_powder_plot_dialog():
        if not state.get('framepeaklist'):
            messagebox.showwarning("No Peaks", "Please run 'Map All Peaks (ROI)' first to accumulate peaks.", parent=root)
            return

        M_active = np.array([
            [state['eps11'], state['eps12']],
            [state['eps21'], state['eps22']]
        ], dtype=np.float64)

        det_shape = (state.get('dety', 128), state.get('detx', 128))

        if powder_win_ref[0] is not None and powder_win_ref[0].winfo_exists():
            powder_win_ref[0].lift()
            powder_win_ref[0].update_plot()
        else:
            powder_win_ref[0] = PowderPlotDialog(
                root,
                state['framepeaklist'],
                rec_samp=state.get('rec_samp', 1.0),
                det_shape=det_shape,
                distortion_matrix=M_active,
                on_range_select_cb=on_powder_range_selected
            )
            
    def run_batch_strain_mapping():
        if state['myset'] is None:
            messagebox.showwarning("No Dataset", "Please load a dataset before generating strain maps.", parent=root)
            return

        candidates = state.get('orientation_candidates', [])
        if not candidates:
            messagebox.showwarning("No Orientation", "Please index an orientation plan first.", parent=root)
            return

        cand = candidates[state.get('selected_candidate_idx', 0)]
        d_theta = np.radians(state.get('template_rot_offset', 0.0))
        cos_t, sin_t = np.cos(d_theta), np.sin(d_theta)

        sim_coords = []
        for spot in cand['sim_spots']:
            qx_b, qy_b = spot[0], spot[1]
            qx_r = cos_t * qx_b - sin_t * qy_b
            qy_r = sin_t * qx_b + cos_t * qy_b
            sim_coords.append([qx_r, qy_r])
        sim_coords = np.array(sim_coords, dtype=np.float64)
        sim_coords = sim_coords[np.linalg.norm(sim_coords, axis=1) > 0.05]

        scany, scanx = state['scany'], state['scanx']

        if state.get('use_map_roi', False):
            roi = state.get('map_roi', {})
            y_start = max(0, roi.get('ymin', 0))
            y_end = min(scany, roi.get('ymax', scany - 1) + 1)
            x_start = max(0, roi.get('xmin', 0))
            x_end = min(scanx, roi.get('xmax', scanx - 1) + 1)
        else:
            y_start, y_end = 0, scany
            x_start, x_end = 0, scanx

        total_pixels = (y_end - y_start) * (x_end - x_start)
        r_cutoff = float(state.get('strain_tolerance', 0.35))
        pars = state['peak_params'].copy()
        use_lin = pars.get('fit_scale_linear', False)

        geom_cx, geom_cy = state['detx'] / 2.0, state['dety'] / 2.0
        rec_samp = state['rec_samp']
        init_cqx, init_cqy = state['cqx'], state['cqy']
        origin_upper = (state['right_origin'] == 'upper')

        map_xx = np.full((scany, scanx), np.nan, dtype=np.float64)
        map_yy = np.full((scany, scanx), np.nan, dtype=np.float64)
        map_xy = np.full((scany, scanx), np.nan, dtype=np.float64)
        map_om = np.full((scany, scanx), np.nan, dtype=np.float64)

        num_workers = max(1, (os.cpu_count() or 4) - 1)
        cancel_flag = [False]
        prog_dlg = StrainMapProgressDialog(root, total_pixels, lambda: cancel_flag.__setitem__(0, True))
        prog_dlg.lbl_info.configure(text=f"Mapping strain ({total_pixels} px, {num_workers} processes)...")

        chunk_size = max(16, total_pixels // (num_workers * 8))
        chunks = []
        curr_chunk = []
        data_4d = state['myset'].data

        for y_idx in range(y_start, y_end):
            for x_idx in range(x_start, x_end):
                frame = data_4d[y_idx, x_idx, :, :].astype(np.float64)
                item = (y_idx, x_idx, frame, pars, use_lin, geom_cx, geom_cy,
                        init_cqx, init_cqy, rec_samp, origin_upper, sim_coords, r_cutoff)
                curr_chunk.append(item)
                if len(curr_chunk) >= chunk_size:
                    chunks.append(curr_chunk)
                    curr_chunk = []
        if curr_chunk:
            chunks.append(curr_chunk)

        def worker_thread():
            start_t = time.time()
            processed = 0

            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(_strain_worker_chunk, chunk) for chunk in chunks]

                for fut in concurrent.futures.as_completed(futures):
                    if cancel_flag[0]:
                        for f in futures:
                            f.cancel()
                        executor.shutdown(wait=False, cancel_futures=True)
                        break

                    chunk_results = fut.result()
                    for py, px, exx, eyy, exy, om in chunk_results:
                        map_xx[py, px] = exx
                        map_yy[py, px] = eyy
                        map_xy[py, px] = exy
                        map_om[py, px] = om

                    processed += len(chunk_results)
                    pct = int((processed / total_pixels) * 100)
                    root.after(0, lambda p=processed, t=f"Pixel: {processed}/{total_pixels} ({pct}%)": prog_dlg.update_progress(p, t))

            def on_finish():
                if not cancel_flag[0]:
                    prog_dlg.destroy()
                    state['strain_maps']['eps_xx'] = map_xx
                    state['strain_maps']['eps_yy'] = map_yy
                    state['strain_maps']['eps_xy'] = map_xy
                    state['strain_maps']['omega'] = map_om

                    btn_show_map_win.configure(fg_color="#00897b", hover_color="#00695c")
                    open_mapping_window()
                    messagebox.showinfo(
                        "Complete", 
                        f"Mapped strain across {total_pixels} pixels in {time.time() - start_t:.1f}s ({num_workers} processes).", 
                        parent=root
                    )

            root.after(0, on_finish)

        threading.Thread(target=worker_thread, daemon=True).start()

    def run_batch_orientation_mapping():
        if state['myset'] is None:
            messagebox.showwarning("No Dataset", "Please load a dataset before generating orientation maps.", parent=root)
            return
        if not state['acom_mgr'].plan_ready or state['acom_mgr'].crystal is None:
            messagebox.showwarning("No Library", "Please load or generate an orientation plan in the Indexing tab.", parent=root)
            return

        scany, scanx = state['scany'], state['scanx']

        if state.get('use_map_roi', False):
            roi = state.get('map_roi', {})
            y_start = max(0, roi.get('ymin', 0))
            y_end = min(scany, roi.get('ymax', scany - 1) + 1)
            x_start = max(0, roi.get('xmin', 0))
            x_end = min(scanx, roi.get('xmax', scanx - 1) + 1)
        else:
            y_start, y_end = 0, scany
            x_start, x_end = 0, scanx

        total_pixels = (y_end - y_start) * (x_end - x_start)
        pars = state['peak_params'].copy()
        use_lin = pars.get('fit_scale_linear', False)

        geom_cx, geom_cy = state['detx'] / 2.0, state['dety'] / 2.0
        rec_samp = state['rec_samp']
        init_cqx, init_cqy = state['cqx'], state['cqy']
        origin_upper = (state['right_origin'] == 'upper')
        M_active = np.array([[state['eps11'], state['eps12']], [state['eps21'], state['eps22']]], dtype=np.float64)

        map_angle = np.full((scany, scanx), np.nan, dtype=np.float64)
        map_corr = np.zeros((scany, scanx), dtype=np.float64)
        map_za = np.full((scany, scanx), np.nan, dtype=np.float64)
        map_R = np.full((scany, scanx, 3, 3), np.nan, dtype=np.float64)

        num_workers = max(1, (os.cpu_count() or 4) - 1)
        cancel_flag = [False]
        prog_dlg = StrainMapProgressDialog(root, total_pixels, lambda: cancel_flag.__setitem__(0, True))
        prog_dlg.lbl_info.configure(text=f"Batch orientation indexing ({total_pixels} px, {num_workers} processes)...")

        chunk_size = max(16, total_pixels // (num_workers * 8))
        chunks = []
        curr_chunk = []
        data_4d = state['myset'].data
        acom_mgr = state['acom_mgr']

        for y_idx in range(y_start, y_end):
            for x_idx in range(x_start, x_end):
                frame = data_4d[y_idx, x_idx, :, :].astype(np.float64)
                item = (y_idx, x_idx, frame, pars, use_lin, geom_cx, geom_cy,
                        init_cqx, init_cqy, rec_samp, origin_upper, M_active, acom_mgr)
                curr_chunk.append(item)
                if len(curr_chunk) >= chunk_size:
                    chunks.append(curr_chunk)
                    curr_chunk = []
        if curr_chunk:
            chunks.append(curr_chunk)

        def worker_thread():
            start_t = time.time()
            processed = 0
            fitted_count = 0

            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(_orient_worker_chunk, chunk) for chunk in chunks]

                for fut in concurrent.futures.as_completed(futures):
                    if cancel_flag[0]:
                        for f in futures:
                            f.cancel()
                        executor.shutdown(wait=False, cancel_futures=True)
                        break

                    chunk_results = fut.result()
                    for py, px, corr_val, angle_val, za_id, R_mat in chunk_results:
                        map_corr[py, px] = corr_val
                        map_angle[py, px] = angle_val
                        map_za[py, px] = za_id
                        if R_mat is not None:
                            map_R[py, px] = R_mat
                            fitted_count += 1

                    processed += len(chunk_results)
                    pct = int((processed / total_pixels) * 100)
                    root.after(0, lambda p=processed, t=f"Pixel: {processed}/{total_pixels} ({pct}%) | Matched: {fitted_count}": prog_dlg.update_progress(p, t))

            def on_finish():
                if not cancel_flag[0]:
                    prog_dlg.destroy()
                    state['orientation_maps']['R_matrices'] = map_R
                    state['orientation_maps']['corr'] = map_corr
                    state['orientation_maps']['angle'] = map_angle
                    state['orientation_maps']['zone_axis_id'] = map_za

                    btn_show_orient_win.configure(fg_color="#6a1b9a", hover_color="#4a148c")
                    open_orient_mapping_window()
                    messagebox.showinfo(
                        "Complete", 
                        f"Indexed orientation maps across {total_pixels} pixels in {time.time() - start_t:.1f}s ({fitted_count} matched, {num_workers} processes).", 
                        parent=root
                    )

            root.after(0, on_finish)

        threading.Thread(target=worker_thread, daemon=True).start()

    def run_batch_peak_detection():
        if state['myset'] is None:
            messagebox.showwarning("No Dataset", "Please load a dataset before running batch peak detection.", parent=root)
            return

        scany, scanx = state['scany'], state['scanx']

        if state.get('use_map_roi', False):
            roi = state.get('map_roi', {})
            y_start = max(0, roi.get('ymin', 0))
            y_end = min(scany, roi.get('ymax', scany - 1) + 1)
            x_start = max(0, roi.get('xmin', 0))
            x_end = min(scanx, roi.get('xmax', scanx - 1) + 1)
        else:
            y_start, y_end = 0, scany
            x_start, x_end = 0, scanx

        total_pixels = (y_end - y_start) * (x_end - x_start)
        pars = state['peak_params'].copy()
        use_lin = pars.get('fit_scale_linear', False)
        origin = state.get('right_origin', 'lower')

        num_workers = max(1, (os.cpu_count() or 4) - 1)
        cancel_flag = [False]
        prog_dlg = StrainMapProgressDialog(root, total_pixels, lambda: cancel_flag.__setitem__(0, True))
        prog_dlg.lbl_info.configure(text=f"Detecting peaks ({total_pixels} px, {num_workers} processes)...")

        # Partition work into chunks to minimize IPC serialization overhead
        chunk_size = max(16, total_pixels // (num_workers * 8))
        chunks = []
        curr_chunk = []
        data_4d = state['myset'].data

        for y_idx in range(y_start, y_end):
            for x_idx in range(x_start, x_end):
                frame = data_4d[y_idx, x_idx, :, :].astype(np.float64)
                item = (y_idx, x_idx, frame, pars, origin, use_lin)
                curr_chunk.append(item)
                if len(curr_chunk) >= chunk_size:
                    chunks.append(curr_chunk)
                    curr_chunk = []
        if curr_chunk:
            chunks.append(curr_chunk)

        def worker_thread():
            start_t = time.time()
            processed = 0
            gathered_peaks = []

            with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
                futures = [executor.submit(_batch_peak_worker_chunk, chunk) for chunk in chunks]

                for fut in concurrent.futures.as_completed(futures):
                    if cancel_flag[0]:
                        for f in futures:
                            f.cancel()
                        executor.shutdown(wait=False, cancel_futures=True)
                        break

                    chunk_results = fut.result()
                    gathered_peaks.extend(chunk_results)
                    processed += len(chunk_results)
                    pct = int((processed / total_pixels) * 100)
                    root.after(0, lambda p=processed, t=f"Pixel: {processed}/{total_pixels} ({pct}%)": prog_dlg.update_progress(p, t))

            def on_finish():
                if not cancel_flag[0]:
                    prog_dlg.destroy()
                    # Sort to preserve raster scan order (row i, column j)
                    gathered_peaks.sort(key=lambda d: (d['i'], d['j']))
                    state['framepeaklist'] = gathered_peaks

                    total_peaks_found = sum(len(d.get('q', d.get('x', []))) for d in gathered_peaks)
                    elapsed = time.time() - start_t
                    lbl_status.configure(text=f"Batch peak detection complete: {total_peaks_found} total peaks found across {len(gathered_peaks)} scan pixels.")
                    messagebox.showinfo(
                        "Detection Complete",
                        f"Processed {len(gathered_peaks)} scan pixels in {elapsed:.1f}s ({num_workers} processes).\nTotal Bragg peaks found: {total_peaks_found}",
                        parent=root
                    )
                    
            root.after(0, on_finish)

        threading.Thread(target=worker_thread, daemon=True).start()
        
    def compute_virtual_image(center_qx, center_qy, radius_in, radius_out, mode='Sum'):
        dety, detx = state['myset'].dim[2], state['myset'].dim[3]
        qy_2d, qx_2d = np.mgrid[:dety, :detx]
        dist_sq = (qx_2d - center_qx)**2 + (qy_2d - center_qy)**2
        mask = (dist_sq >= radius_in**2) & (dist_sq <= radius_out**2)
        
        if np.any(mask):
            masked_data = state['myset'].data[:, :, mask]
            if mode == 'Sum': vimage = np.sum(masked_data, axis=-1)
            elif mode == 'Variance': vimage = np.var(masked_data, axis=-1)
            elif mode == 'Fluctuation':
                mean_val = np.mean(masked_data, axis=-1)
                var_val = np.var(masked_data, axis=-1)
                vimage = np.where(mean_val > 1e-6, var_val / (mean_val**2 + 1e-6), 0.0)
            elif mode in ['COM Mag', 'COM Azimuth']:
                rel_qx = (qx_2d - center_qx)[mask]
                rel_qy = (qy_2d - center_qy)[mask]
                I_total = np.maximum(1e-6, np.sum(masked_data, axis=-1))
                com_x = np.sum(masked_data * rel_qx, axis=-1) / I_total
                com_y = np.sum(masked_data * rel_qy, axis=-1) / I_total
                vimage = np.hypot(com_x, com_y) * state['rec_samp'] if mode == 'COM Mag' else np.arctan2(com_y, com_x)
            else: vimage = np.sum(masked_data, axis=-1)
        else:
            vimage = np.zeros((state['scany'], state['scanx']))
        return vimage, mask

    def extract_avg_diffraction(center_y, center_x, radius_y, radius_x):
        y_min = max(0, center_y - radius_y)
        y_max = min(state['scany'], center_y + radius_y + 1)
        x_min = max(0, center_x - radius_x)
        x_max = min(state['scanx'], center_x + radius_x + 1)
        
        raw_slice = state['myset'].data[y_min:y_max, x_min:x_max, :, :].astype(np.float64)
        n_frames = (2 * radius_x + 1) * (2 * radius_y + 1)
        norm_avg = np.sum(raw_slice, axis=(0, 1)) / max(1, n_frames)
        state['norm_img'] = norm_avg

        avg_slice = norm_avg.copy()
        if state['bg_sub']:
            det_x = state['myset'].dim[3]
            ksize = int(np.clip(state['bg_kernel'], 2, max(2, det_x // 4)))
            if ksize % 2 == 0: ksize += 1
            avg_slice = np.maximum(0.0, avg_slice - median_filter(avg_slice, size=ksize))

        return np.log(avg_slice + 1.0) if state['right_scale'] == 'log' else avg_slice

    def apply_contrast(data, clip_percentiles, gamma):
        valid_vals = data[~np.isnan(data)]
        if valid_vals.size == 0:
            return np.zeros_like(data)
        vmin, vmax = np.percentile(valid_vals, clip_percentiles)
        if np.isclose(vmin, vmax):
            vmin = np.min(valid_vals)
            vmax = np.max(valid_vals)
            if np.isclose(vmin, vmax):
                return np.full_like(data, 0.5)
        norm = np.clip((data - vmin) / (vmax - vmin), 0.0, 1.0)
        return np.power(norm, gamma)

    def update_left_colorbar(vmin_real, vmax_real, is_strain_mode, unit_label=""):
        if is_strain_mode:
            if artists['cbar1'] is None:
                if divider_cache[0] is None:
                    divider_cache[0] = make_axes_locatable(ax1)
                cax = divider_cache[0].append_axes("right", size="3.5%", pad=0.08)
                norm = mcolors.Normalize(vmin=vmin_real, vmax=vmax_real)
                artists['cbar1'] = fig.colorbar(artists['im1'], cax=cax, norm=norm)
            else:
                artists['cbar1'].ax.set_visible(True)
                norm = mcolors.Normalize(vmin=vmin_real, vmax=vmax_real)
                artists['cbar1'].mappable.set_norm(norm)
                artists['cbar1'].mappable.set_clim(vmin_real, vmax_real)

            lbl_col = 'white' if state.get('theme', 'dark') == 'dark' else 'black'
            artists['cbar1'].set_label(unit_label, color=lbl_col, fontsize=9)
            artists['cbar1'].ax.tick_params(labelsize=8, colors=lbl_col)
            artists['cbar1'].update_normal(artists['im1'])
        else:
            if artists['cbar1'] is not None:
                artists['cbar1'].ax.set_visible(False)

    def update_left():
        if state['myset'] is None or 'data_left_raw' not in state or state['data_left_raw'] is None:
            return

        raw_data = state['data_left_raw']
        curr_cmap = plt.get_cmap(state['left_cmap']).copy()
        curr_cmap.set_bad(color='black', alpha=0.0)
        artists['im1'].set_cmap(curr_cmap)

        c_low = float(var_l_clip_low.get())
        c_high = float(var_l_clip_high.get())
        gamma = float(var_l_gamma.get())
        processed = apply_contrast(raw_data, (c_low, c_high), gamma)
        artists['im1'].set_clim(0.0, 1.0)
        artists['im1'].set_data(processed)
        update_left_colorbar(0.0, 1.0, is_strain_mode=False)

        update_image_statistics()
        canvas.draw_idle()

    def update_right():
        if state['myset'] is None: return
        c_low, c_high = float(var_r_clip_low.get()), float(var_r_clip_high.get())
        gamma = float(var_r_gamma.get())
        processed = apply_contrast(state['data_right_raw'], (c_low, c_high), gamma)
        artists['im2'].set_cmap(state['right_cmap'])
        artists['im2'].set_data(processed)
        update_image_statistics()
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
        y_box_min = (state['scany'] - 1 - state['y'] - state['dy'] - 0.5) * state['re_samp'] if state['left_origin'] == 'upper' else (state['y'] - state['dy'] - 0.5) * state['re_samp']
        artists['roi_rect'].set_bounds((state['x'] - state['dx'] - 0.5) * state['re_samp'], y_box_min, (2*state['dx'] + 1) * state['re_samp'], (2*state['dy'] + 1) * state['re_samp'])
        state['data_right_raw'] = extract_avg_diffraction(state['y'], state['x'], state['dy'], state['dx'])
        ax2.set_title(f"Diffraction Frame ({state['y']}, {state['x']})", pad=15)
        update_right()

    def rebuild_plot():
        ax1.clear(); ax2.clear()
        artists['cbar1'] = None

        if state['myset'] is None:
            dummy_left = np.zeros((256, 256), dtype=np.float32)
            dummy_right = np.zeros((128, 128), dtype=np.float32)
            artists['im1'] = ax1.imshow(dummy_left, cmap=state['left_cmap'], extent=[0, 100, 0, 100], origin=state['left_origin'])
            ax1.set_title("Virtual Image [No Dataset Loaded]", pad=15)
            ax1.set_xlabel("x (nm)"); ax1.set_ylabel("y (nm)")

            artists['im2'] = ax2.imshow(dummy_right, cmap=state['right_cmap'], extent=[-5, 5, -5, 5], origin=state['right_origin'])
            ax2.set_title("Diffraction Frame [No Dataset Loaded]", pad=15)
            ax2.set_xlabel("q_x (1/nm)"); ax2.set_ylabel("q_y (1/nm)")
            canvas.draw()
            return

        scany, scanx = state['myset'].dim[0], state['myset'].dim[1]
        dety, detx = state['myset'].dim[2], state['myset'].dim[3]
        state['scany'], state['scanx'] = scany, scanx
        state['dety'], state['detx'] = dety, detx

        if state['x'] == 0 and state['y'] == 0:
            state['x'], state['y'] = scanx // 2, scany // 2
        if state['cqx'] == 0 and state['cqy'] == 0:
            state['cqx'], state['cqy'] = detx // 2, dety // 2

        extent_left = [0, scanx * state['re_samp'], 0, scany * state['re_samp']]
        extent_right = get_reciprocal_extent(state['cqx'], state['cqy'])

        state['data_left_raw'], _ = compute_virtual_image(state['cqx'], state['cqy'], state['r_in'], state['r_out'], mode=state['calc_mode'])
        state['data_right_raw'] = extract_avg_diffraction(state['y'], state['x'], state['dy'], state['dx'])

        artists['im1'] = ax1.imshow(state['data_left_raw'], cmap=state['left_cmap'], origin=state['left_origin'], extent=extent_left)
        ax1.set_title(f"Virtual Image [{state['calc_mode']}]", pad=15)
        ax1.set_xlabel("x (nm)"); ax1.set_ylabel("y (nm)")

        ax1.secondary_xaxis('top', functions=(lambda val: val / state['re_samp'], lambda val: val * state['re_samp'])).set_xlabel("x (pixels)", labelpad=4)
        ax1.secondary_yaxis('right', functions=(lambda val: val / state['re_samp'], lambda val: val * state['re_samp'])).set_ylabel("y (pixels)", labelpad=4)

        px_x = state['x'] * state['re_samp']
        px_y = (state['scany'] - 1 - state['y']) * state['re_samp'] if state['left_origin'] == 'upper' else state['y'] * state['re_samp']
        artists['v_line'] = ax1.axvline(px_x, color='yellow', lw=1, zorder=3)
        artists['h_line'] = ax1.axhline(px_y, color='yellow', lw=1, zorder=3)

        y_box_min = (state['scany'] - 1 - state['y'] - state['dy'] - 0.5) * state['re_samp'] if state['left_origin'] == 'upper' else (state['y'] - state['dy'] - 0.5) * state['re_samp']
        artists['roi_rect'] = Rectangle(((state['x'] - state['dx'] - 0.5) * state['re_samp'], y_box_min), (2*state['dx'] + 1) * state['re_samp'], (2*state['dy'] + 1) * state['re_samp'],
                                        linewidth=1, edgecolor='yellow', facecolor='none', linestyle='--', zorder=4)
        ax1.add_patch(artists['roi_rect'])

        artists['im2'] = ax2.imshow(apply_contrast(state['data_right_raw'], (0, 100), 1.0), cmap=state['right_cmap'], origin=state['right_origin'], extent=extent_right)
        ax2.set_title(f"Diffraction Frame ({state['y']}, {state['x']})", pad=15)
        ax2.set_xlabel("q_x (1/nm)"); ax2.set_ylabel("q_y (1/nm)")

        ax2.secondary_xaxis('top', functions=(lambda val: val / state['rec_samp'] + state['cqx'], lambda val: (val - state['cqx']) * state['rec_samp'])).set_xlabel("q_x (pixels)", labelpad=4)
        ax2.secondary_yaxis('right', functions=(lambda val: val / state['rec_samp'] + state['cqy'], lambda val: (val - state['cqy']) * state['rec_samp'])).set_ylabel("q_y (pixels)", labelpad=4)

        artists['center_marker'] = ax2.plot(0, 0, 'rx', markersize=8, zorder=5)[0]
        artists['circle_in'] = Circle((0, 0), state['r_in'] * state['rec_samp'], color='yellow', fill=False, linestyle='--', linewidth=1.5, zorder=4)
        artists['circle_out'] = Circle((0, 0), state['r_out'] * state['rec_samp'], color='yellow', fill=False, linestyle='-', linewidth=1.5, zorder=4)
        ax2.add_patch(artists['circle_in'])
        ax2.add_patch(artists['circle_out'])

        artists['ruler1_line'] = ax1.plot([], [], color='#ff00ff', linestyle='--', lw=2.0, marker='o', markersize=5, zorder=10)[0]
        artists['ruler1_text'] = ax1.text(0.03, 0.95, "", transform=ax1.transAxes, color='#ff00ff', fontsize=9, zorder=11,
                                          bbox=dict(boxstyle="round,pad=0.3", fc="black", ec="#ff00ff", alpha=0.85))
        artists['ruler1_text'].set_visible(ruler1_state['active'])

        artists['ruler2_line'] = ax2.plot([], [], color='#00ffff', linestyle='--', lw=2.0, marker='o', markersize=5, zorder=10)[0]
        artists['ruler2_text'] = ax2.text(0.03, 0.95, "", transform=ax2.transAxes, color='#00ffff', fontsize=9, zorder=11,
                                          bbox=dict(boxstyle="round,pad=0.3", fc="black", ec="#00ffff", alpha=0.85))
        artists['ruler2_text'].set_visible(ruler2_state['active'])

        artists['stats_box_left'] = ax1.text(0.03, 0.05, "", transform=ax1.transAxes, color='white', fontsize=8, fontfamily="monospace", zorder=11,
                                             bbox=dict(boxstyle="round,pad=0.4", fc="black", ec="#76ff03", alpha=0.85))
        artists['stats_box_left'].set_visible(False)

        artists['stats_box_right'] = ax2.text(0.03, 0.05, "", transform=ax2.transAxes, color='white', fontsize=8, fontfamily="monospace", zorder=11,
                                              bbox=dict(boxstyle="round,pad=0.4", fc="black", ec="#00e5ff", alpha=0.85))
        artists['stats_box_right'].set_visible(False)

        redraw_ring_markers()
        redraw_peak_markers()
        redraw_template_spots()
        update_map_roi_visual()
        update_image_statistics()
        update_left()

        next_idx, _ = get_next_filename()
        btn_save.configure(text=f"Export Fig #{next_idx:03d}")
        sync_target_dict()
        canvas.draw()

    def open_config_dialog_action():
        nonlocal config
        dialog = ConfigDialog(root, config)
        root.wait_window(dialog)
        if dialog.updated:
            config = load_config()
            ctk.set_appearance_mode(config.get("theme", "dark"))
            state['left_cmap'], state['right_cmap'] = config["left_cmap"], config["right_cmap"]
            state['left_origin'], state['right_origin'] = config.get("left_origin", "lower"), config.get("right_origin", "lower")
            state['right_scale'] = config.get("right_scale", "log")
            state['eps11'], state['eps12'] = config.get("eps11", 1.0), config.get("eps12", 0.0)
            state['eps21'], state['eps22'] = config.get("eps21", 0.0), config.get("eps22", 1.0)
            state['bg_kernel'] = config.get("bg_kernel", 15)
            state['ring_presets'] = config.get("ring_presets", DEFAULT_RING_PRESETS)
            state['larmor_offset_deg'] = float(config.get("larmor_offset_deg", 0.0))  # <-- Sync to state
            
            var_l_cmap.set(config["left_cmap"])
            var_r_cmap.set(config["right_cmap"])
            var_dx.set(str(config["dx"]))
            var_dy.set(str(config["dy"]))
            var_rin.set(str(config["r_in"]))
            var_rout.set(str(config["r_out"]))
            
            # If a selective mapping window is currently open, refresh its view with the new offset
            if selective_win_ref[0] is not None and selective_win_ref[0].winfo_exists():
                selective_win_ref[0].var_larmor_deg.set(state['larmor_offset_deg'])
                selective_win_ref[0].lbl_larmor.configure(text=f"{state['larmor_offset_deg']:+.1f}°")
                selective_win_ref[0].update_display()
                
            if state['myset']: rebuild_plot()
            
    def open_file_dialog_action():
        dialog = FileImportDialog(root, config)
        root.wait_window(dialog)
        if dialog.result:
            res = dialog.result
            state['path'], state['filebasename'], state['filesuffix'] = res['path'], res['filebasename'], res['filesuffix']
            state['format_type'] = res['format_type']
            state['total_intensity_map'] = None
            config['bin_scan_y'], config['bin_scan_x'] = res['bin_scan']
            config['bin_det_y'], config['bin_det_x'] = res['bin_det']
            save_config(config)

            try:
                new_set = nbed.pyNBED()
                args = {'bin_scan': res['bin_scan'], 'bin_det': res['bin_det'], 'scan_shape': res['scan_shape']}
                if res['is_zipfile']: args['is_zipfile'] = True
                new_set.LoadFile(res['full_path'], type=state['format_type'], **args)
                state['myset'] = new_set

                if hasattr(new_set, 'PreparePeakDetectionPars'):
                    state['peak_params'] = new_set.PreparePeakDetectionPars(shortcut=state['detector_preset'])
                    state['peak_params']['fit_scale_linear'] = config.get("fit_scale_linear", False)

                if os.path.exists(config["cal_db_path"]):
                    cal_mgr = nbed.MicroscopeCalibrationManager(config["cal_db_path"])
                    samp, qsamp = cal_mgr.get_calibration_from_metadata(new_set.metadata, type=state['format_type'])
                    state['re_samp'] = samp if samp else 1.0 * res['bin_scan'][1]
                    state['rec_samp'] = qsamp if qsamp else 1.0 * res['bin_det'][1]

                rebuild_plot()
                lbl_status.configure(text=f"Loaded: {res['filebasename']}{res['filesuffix']} [{state['re_samp']:.4g} nm/px, {state['rec_samp']:.4g} 1/nm/px]")
            except Exception as e:
                messagebox.showerror("Loading Error", str(e))

    def manual_sampling_dialog_action():
        if state['myset'] is None:
            messagebox.showwarning("No Dataset", "Please load a dataset before calibrating pixel sampling.", parent=root)
            return

        dlg = ManualSamplingDialog(root, state['re_samp'], state['rec_samp'])
        root.wait_window(dlg)
        
        if dlg.result:
            state['re_samp'] = float(dlg.result['re_samp'])
            state['rec_samp'] = float(dlg.result['rec_samp'])
            
            sync_target_dict()
            rebuild_plot()
            
            lbl_status.configure(
                text=f"Sampling Updated: {state['re_samp']:.5g} nm/px (Real) | {state['rec_samp']:.5g} 1/nm/px (Recip)"
            )
            messagebox.showinfo(
                "Calibration Applied",
                f"Calibrations updated:\nReal Space: {state['re_samp']:.5g} nm/px\nReciprocal Space: {state['rec_samp']:.5g} 1/nm/px",
                parent=root
            )

    menu_bar = tk.Menu(root)
    file_menu = tk.Menu(menu_bar, tearoff=0)
    file_menu.add_command(label="Open Dataset...", command=open_file_dialog_action)
    file_menu.add_command(label="Calibrate Sampling...", command=manual_sampling_dialog_action)
    file_menu.add_command(label="View Metadata...", command=lambda: root.wait_window(MetadataViewerDialog(root, state['myset'].metadata, f"{state['filebasename']}{state['filesuffix']}")) if state['myset'] else None)
    file_menu.add_separator()
    file_menu.add_command(label="Settings...", command=open_config_dialog_action)
    file_menu.add_command(label="Exit", command=root.quit)
    menu_bar.add_cascade(label="File", menu=file_menu)
    root.configure(menu=menu_bar)

    ctx_menu = tk.Menu(root, tearoff=0)
    def rebuild_context_menu():
        ctx_menu.delete(0, tk.END)
        preset_sub = tk.Menu(ctx_menu, tearoff=0)
        for key in state['ring_presets'].keys():
            preset_sub.add_command(label=key, command=lambda k=key: apply_ring_preset_by_key(k))
        ctx_menu.add_cascade(label="🧊 Add Ring Preset", menu=preset_sub)
        ctx_menu.add_separator()
        ctx_menu.add_command(label="🗑️ Clear Ring Markers", command=clear_ring_markers_cmd)

    top_bar = ctk.CTkFrame(root, height=45, corner_radius=0)
    top_bar.pack(side=tk.TOP, fill=tk.X, padx=0, pady=0)

    btn_open = ctk.CTkButton(top_bar, text="📁 Open Dataset...", width=130, command=open_file_dialog_action)
    btn_open.pack(side=tk.LEFT, padx=8, pady=8)

    btn_config = ctk.CTkButton(top_bar, text="⚙️ Settings...", width=110, fg_color="transparent", border_width=1, command=open_config_dialog_action)
    btn_config.pack(side=tk.LEFT, padx=4, pady=8)

    def reset_left_view():
        if state['myset'] is None: return
        scanx, scany = state['scanx'], state['scany']
        ax1.set_xlim(0, scanx * state['re_samp'])
        ax1.set_ylim(scany * state['re_samp'], 0) if state['left_origin'] == 'upper' else ax1.set_ylim(0, scany * state['re_samp'])
        canvas.draw_idle()

    def reset_right_view():
        if state['myset'] is None: return
        extent_right = get_reciprocal_extent(state['cqx'], state['cqy'])
        ax2.set_xlim(extent_right[0], extent_right[1])
        ax2.set_ylim(extent_right[2], extent_right[3])
        canvas.draw_idle()

    ctk.CTkButton(top_bar, text="↺ Reset Real", width=95, fg_color="#37474f", hover_color="#263238", command=reset_left_view).pack(side=tk.LEFT, padx=4, pady=8)
    ctk.CTkButton(top_bar, text="↺ Reset Recip", width=95, fg_color="#37474f", hover_color="#263238", command=reset_right_view).pack(side=tk.LEFT, padx=4, pady=8)

    var_show_stats = tk.BooleanVar(value=state['show_stats'])
    def on_toggle_stats():
        state['show_stats'] = var_show_stats.get()
        update_image_statistics()
        canvas.draw_idle()

    chk_stats = ctk.CTkSwitch(
        top_bar, 
        text="📊 Image Stats", 
        variable=var_show_stats, 
        command=on_toggle_stats,
        font=("Arial", 11, "bold"),
        width=120
    )
    chk_stats.pack(side=tk.LEFT, padx=(12, 6), pady=8)

    lbl_status = ctk.CTkLabel(top_bar, text="No dataset loaded. Click 'Open Dataset' to start.", font=("Arial", 11, "italic"))
    lbl_status.pack(side=tk.LEFT, padx=12, pady=8)

    ctrl_frame = ctk.CTkFrame(root, height=180, corner_radius=0)
    ctrl_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=8, pady=8)

    canvas_frame = ctk.CTkFrame(root, corner_radius=6)
    canvas_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=8, pady=(4, 4))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))
    fig.patch.set_facecolor('#1a1a1a')
    ax1.set_facecolor('#121212')
    ax2.set_facecolor('#121212')
    fig.subplots_adjust(top=0.88, bottom=0.15, left=0.08, right=0.92, wspace=0.35)

    canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
    canvas_widget = canvas.get_tk_widget()
    canvas_widget.pack(fill=tk.BOTH, expand=True)

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

    var_l_cmap = ctk.StringVar(value=state['left_cmap'])
    def on_l_cmap_select(choice):
        state['left_cmap'] = choice
        config['left_cmap'] = choice
        save_config(config)
        if state['myset']:
            artists['im1'].set_cmap(choice)
            update_left()

    available_cmaps = get_available_display_colormaps()

    cb_l_cmap = ctk.CTkOptionMenu(
        f_left,
        variable=var_l_cmap,
        values=available_cmaps,
        width=110,
        command=on_l_cmap_select
    )
    cb_l_cmap.grid(row=3, column=1, columnspan=2, pady=4)
    

    # 2. Virtual Image Mask Mode
    f_mode = ctk.CTkFrame(ctrl_frame)
    f_mode.pack(side=tk.LEFT, fill=tk.Y, padx=4, pady=4)
    ctk.CTkLabel(f_mode, text="Mask Mode", font=("Arial", 11, "bold")).grid(row=0, column=0, pady=(4,2))

    var_mode = ctk.StringVar(value='Sum')
    modes = ['Sum', 'Variance', 'Fluctuation', 'COM Mag', 'COM Azimuth']

    def on_mode_change():
        chosen = var_mode.get()
        state['calc_mode'] = chosen
        if chosen == 'COM Azimuth':
            state['left_cmap'] = 'hsv'
            var_l_cmap.set('hsv')
        else:
            if state['left_cmap'] == 'coolwarm':
                state['left_cmap'] = 'gray'
                var_l_cmap.set('gray')
        sync_target_dict()
        refresh_detector_mask()

    for i, m in enumerate(modes):
        rb = ctk.CTkRadioButton(f_mode, text=m, value=m, variable=var_mode, command=on_mode_change, font=("Arial", 10), radiobutton_width=14, radiobutton_height=14)
        rb.grid(row=i + 1, column=0, sticky='w', padx=6, pady=2)

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

    var_r_scale = ctk.StringVar(value=state['right_scale'].capitalize())
    def on_r_scale_select(choice):
        state['right_scale'] = choice.lower()
        config['right_scale'] = choice.lower()
        save_config(config)
        refresh_spatial_roi()

    cb_r_scale = ctk.CTkOptionMenu(f_right, variable=var_r_scale, values=["Log", "Linear"], width=90, command=on_r_scale_select)
    cb_r_scale.grid(row=3, column=1, columnspan=2, pady=2)

    var_r_cmap = ctk.StringVar(value=state['right_cmap'])
    def on_r_cmap_select(choice):
        state['right_cmap'] = choice
        config['right_cmap'] = choice
        save_config(config)
        if state['myset']:
            update_right()

    cb_r_cmap = ctk.CTkOptionMenu(
        f_right, 
        variable=var_r_cmap, 
        values=available_cmaps, 
        width=110, 
        command=on_r_cmap_select
    )
    
    cb_r_cmap.grid(row=4, column=1, columnspan=2, pady=2)

    # 4. ROI & Mask Size Parameters
    f_params = ctk.CTkFrame(ctrl_frame)
    f_params.pack(side=tk.LEFT, fill=tk.Y, padx=4, pady=4)
    ctk.CTkLabel(f_params, text="ROI & Mask Radii", font=("Arial", 11, "bold")).grid(row=0, column=0, columnspan=5, pady=(4,2))

    ctk.CTkLabel(f_params, text="dx:").grid(row=1, column=0, sticky='e')
    var_dx = ctk.StringVar(value=str(state['dx']))
    ctk.CTkEntry(f_params, textvariable=var_dx, width=38).grid(row=1, column=1)

    ctk.CTkLabel(f_params, text="dy:").grid(row=1, column=2, sticky='e')
    var_dy = ctk.StringVar(value=str(state['dy']))
    ctk.CTkEntry(f_params, textvariable=var_dy, width=38).grid(row=1, column=3)
    ctk.CTkButton(f_params, text="Set ROI", width=55, command=lambda: [config.__setitem__('dx', int(var_dx.get())), config.__setitem__('dy', int(var_dy.get())), refresh_spatial_roi()]).grid(row=1, column=4, padx=4)

    ctk.CTkLabel(f_params, text="r_in:").grid(row=2, column=0, sticky='e')
    var_rin = ctk.StringVar(value=str(state['r_in']))
    ctk.CTkEntry(f_params, textvariable=var_rin, width=38).grid(row=2, column=1)

    ctk.CTkLabel(f_params, text="r_out:").grid(row=2, column=2, sticky='e')
    var_rout = ctk.StringVar(value=str(state['r_out']))
    ctk.CTkEntry(f_params, textvariable=var_rout, width=38).grid(row=2, column=3)
    ctk.CTkButton(f_params, text="Set Mask", width=55, command=lambda: [config.__setitem__('r_in', float(var_rin.get())), config.__setitem__('r_out', float(var_rout.get())), refresh_detector_mask()]).grid(row=2, column=4, padx=4)

    # 5. Docking Tabs
    tab_tools = ctk.CTkTabview(ctrl_frame, width=540, height=140)
    tab_tools.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=4, pady=2)
    
    tab_tools.add("Centering")
    tab_tools.add("Rulers")
    tab_tools.add("Peak Finding")
    tab_tools.add("Indexing")
    tab_tools.add("Mapping")
    tab_tools.add("Export")

    # TAB 1: Centering
    tab_pr = tab_tools.tab("Centering")
    f_pr_row1 = ctk.CTkFrame(tab_pr, fg_color="transparent")
    f_pr_row1.pack(fill=tk.X, pady=(2, 2))

    ctk.CTkButton(f_pr_row1, text="🎯 Align Beam", width=115, fg_color="#2e7d32", hover_color="#1b5e20", command=align_diffraction_centers).pack(side=tk.LEFT, padx=2)

    var_bg_sub = tk.BooleanVar(value=state['bg_sub'])
    var_bg_kernel = ctk.StringVar(value=str(state['bg_kernel']))

    def on_bg_sub_toggle():
        state['bg_sub'] = var_bg_sub.get()
        try:
            k = int(np.clip(int(var_bg_kernel.get()), 2, max(2, state['myset'].dim[3] // 4 if state['myset'] else 128)))
            state['bg_kernel'] = k
            var_bg_kernel.set(str(k))
        except ValueError:
            state['bg_kernel'] = 15
        sync_target_dict()
        refresh_spatial_roi()

    chk_bg = ctk.CTkCheckBox(f_pr_row1, text="Sub BG", variable=var_bg_sub, command=on_bg_sub_toggle, font=("Arial", 10), width=65)
    chk_bg.pack(side=tk.LEFT, padx=4)

    f_pr_row2 = ctk.CTkFrame(tab_pr, fg_color="transparent")
    f_pr_row2.pack(fill=tk.X, pady=(2, 2))
    ctk.CTkLabel(f_pr_row2, text="BG Filter Kernel k:").pack(side=tk.LEFT, padx=2)
    entry_kernel = ctk.CTkEntry(f_pr_row2, textvariable=var_bg_kernel, width=42)
    entry_kernel.pack(side=tk.LEFT, padx=2)
    var_bg_kernel.trace_add("write", lambda *a: on_bg_sub_toggle() if state['bg_sub'] else None)

    # TAB 2: Rulers & Calibrations
    tab_m = tab_tools.tab("Rulers")
    f_m_top = ctk.CTkFrame(tab_m, fg_color="transparent")
    f_m_top.pack(fill=tk.X, pady=(2, 2))

    btn_r1 = ctk.CTkButton(f_m_top, text="Ruler Real", width=80)
    btn_r2 = ctk.CTkButton(f_m_top, text="Ruler Recip", width=80)
    btn_calib = ctk.CTkButton(
        f_m_top, text="📐 Calib", width=65, fg_color="#f57c00", hover_color="#e65100", 
        command=lambda: root.wait_window(
            DistortionCalibDialog(
                root, state['ring_presets'], 
                np.array([[state['eps11'], state['eps12']], [state['eps21'], state['eps22']]]), 
                draw_calib_elliptical_rings, clear_calib_rings, config
            )
        )
    )

    def toggle_ruler1():
        if state['myset'] is None: return
        ruler1_state['active'] = not ruler1_state['active']
        ruler1_state['p1'], ruler1_state['p2'], ruler1_state['length'] = None, None, 0.0
        btn_r1.configure(text="Real ON" if ruler1_state['active'] else "Ruler Real", 
                         fg_color="#c2185b" if ruler1_state['active'] else ctk.ThemeManager.theme["CTkButton"]["fg_color"])
        if ruler1_state['active']:
            artists['ruler1_text'].set_text("Shift+Click 2 points")
            artists['ruler1_text'].set_visible(True)
        else:
            artists['ruler1_text'].set_visible(False)
            artists['ruler1_line'].set_data([], [])
        canvas.draw_idle()

    def toggle_ruler2():
        if state['myset'] is None: return
        ruler2_state['active'] = not ruler2_state['active']
        ruler2_state['p1'], ruler2_state['p2'], ruler2_state['dq'] = None, None, 0.0
        btn_r2.configure(text="Recip ON" if ruler2_state['active'] else "Ruler Recip", 
                         fg_color="#00838f" if ruler2_state['active'] else ctk.ThemeManager.theme["CTkButton"]["fg_color"])
        if ruler2_state['active']:
            artists['ruler2_text'].set_text("Shift+Click 2 points")
            artists['ruler2_text'].set_visible(True)
        else:
            artists['ruler2_text'].set_visible(False)
            artists['ruler2_line'].set_data([], [])
        canvas.draw_idle()

    btn_r1.configure(command=toggle_ruler1)
    btn_r2.configure(command=toggle_ruler2)
    btn_r1.pack(side=tk.LEFT, padx=3)
    btn_r2.pack(side=tk.LEFT, padx=3)
    btn_calib.pack(side=tk.LEFT, padx=3)

    f_rings = ctk.CTkFrame(tab_m, fg_color="transparent")
    f_rings.pack(fill=tk.X, pady=(2, 2))
    ctk.CTkLabel(f_rings, text="q:").pack(side=tk.LEFT, padx=2)
    var_ring_q = ctk.StringVar(value="2.571")
    ctk.CTkEntry(f_rings, textvariable=var_ring_q, width=48).pack(side=tk.LEFT, padx=2)

    def add_ring_marker_cmd():
        try:
            val_q = float(var_ring_q.get())
            if val_q <= 0: raise ValueError()
            state['ring_markers'].append(val_q)
            sync_target_dict()
            redraw_ring_markers()
            canvas.draw_idle()
        except ValueError:
            messagebox.showerror("Error", "Enter positive numeric q radius in 1/nm.")

    def clear_ring_markers_cmd():
        state['ring_markers'].clear()
        sync_target_dict()
        redraw_ring_markers()
        canvas.draw_idle()

    ctk.CTkButton(f_rings, text="+", width=28, command=add_ring_marker_cmd).pack(side=tk.LEFT, padx=2)
    
    default_preset = list(state['ring_presets'].keys())[0] if state['ring_presets'] else ""
    cb_preset_menu = ctk.CTkOptionMenu(
        f_rings, variable=ctk.StringVar(value=default_preset), 
        values=list(state['ring_presets'].keys()), width=100, command=apply_ring_preset_by_key, font=("Arial", 10)
    )
    cb_preset_menu.pack(side=tk.LEFT, padx=2)
    ctk.CTkButton(f_rings, text="🗑️", width=32, fg_color="#d32f2f", hover_color="#9a0007", command=clear_ring_markers_cmd).pack(side=tk.LEFT, padx=2)

    # TAB 3: Peak Finding
    tab_peaks = tab_tools.tab("Peak Finding")
    f_pk_r1 = ctk.CTkFrame(tab_peaks, fg_color="transparent")
    f_pk_r1.pack(fill=tk.X, pady=(3, 3))

    var_det_preset = ctk.StringVar(value=state['detector_preset'])
    def on_det_preset_change(choice):
        state['detector_preset'] = choice
        config['detector_preset'] = choice
        save_config(config)
        if state['myset'] is not None and hasattr(state['myset'], 'PreparePeakDetectionPars'):
            state['peak_params'] = state['myset'].PreparePeakDetectionPars(shortcut=choice)
            state['peak_params']['fit_scale_linear'] = config.get("fit_scale_linear", False)

    cb_det_preset = ctk.CTkOptionMenu(f_pk_r1, variable=var_det_preset, values=["ELA", "ARINA", "EMPAD"], width=85, command=on_det_preset_change)
    cb_det_preset.pack(side=tk.LEFT, padx=2)

    btn_fit_pks = ctk.CTkButton(f_pk_r1, text="🔍 Fit", width=55, fg_color="#00897b", hover_color="#00695c", command=run_peak_detection)
    btn_fit_pks.pack(side=tk.LEFT, padx=2)

    btn_pks_cfg = ctk.CTkButton(f_pk_r1, text="⚙️ Params", width=65, fg_color="#37474f", hover_color="#263238", command=open_peak_params_dialog)
    btn_pks_cfg.pack(side=tk.LEFT, padx=2)

    btn_pks_clr = ctk.CTkButton(f_pk_r1, text="🗑️ Clear", width=55, fg_color="#c62828", hover_color="#8e0000", command=clear_peak_detection)
    btn_pks_clr.pack(side=tk.LEFT, padx=2)

    f_pk_r3 = ctk.CTkFrame(tab_peaks, fg_color="transparent")
    f_pk_r3.pack(fill=tk.X, pady=(3, 2))

    lbl_pk_slider = ctk.CTkLabel(
        f_pk_r3, 
        text=f"Top Peaks: {int(state['peak_params'].get('peak_keep_percent', 100))}%", 
        width=100, 
        font=("Arial", 10), 
        anchor='w'
    )
    lbl_pk_slider.pack(side=tk.LEFT, padx=(4, 2))

    def on_quick_pk_slider(val):
        pct = int(round(float(val)))
        state['peak_params']['peak_keep_percent'] = float(pct)
        lbl_pk_slider.configure(text=f"Top Peaks: {pct}%")
        run_peak_detection()

    slider_quick_pk = ctk.CTkSlider(
        f_pk_r3, 
        from_=5.0, 
        to=100.0, 
        number_of_steps=95, 
        width=170, 
        command=on_quick_pk_slider
    )
    slider_quick_pk.set(state['peak_params'].get('peak_keep_percent', 100.0))
    slider_quick_pk.pack(side=tk.LEFT, padx=3)

    # TAB 4: Indexing
    tab_orient = tab_tools.tab("Indexing")

    def open_library_manager():
        TemplateLibraryDialog(
            root,
            state['acom_mgr'],
            on_library_changed=update_crystal_badge
        )

    f_or_r1 = ctk.CTkFrame(tab_orient, fg_color="transparent")
    f_or_r1.pack(fill=tk.X, pady=(2, 2))

    btn_cryst_lib = ctk.CTkButton(
        f_or_r1,
        text="📚 Library: None",
        font=("Arial", 11, "bold"),
        fg_color="#37474f",
        hover_color="#263238",
        text_color="#cfd8dc",
        corner_radius=6,
        command=open_library_manager
    )
    btn_cryst_lib.pack(side=tk.LEFT, padx=3)

    def update_crystal_badge():
        acom = state.get('acom_mgr')
        if acom is not None and acom.plan_ready and acom.crystal is not None:
            cif_name = ""
            if acom.cif_path:
                cif_name = os.path.splitext(os.path.basename(acom.cif_path))[0]
            elif acom.library_path:
                cif_name = os.path.splitext(os.path.basename(acom.library_path))[0]
                if cif_name.endswith('.pkl'):
                    cif_name = os.path.splitext(cif_name)[0]
            else:
                cif_name = "Active"

            n_templates = 0
            if hasattr(acom.crystal, 'orientation_zone_axes'):
                za = getattr(acom.crystal, 'orientation_zone_axes')
                n_templates = len(za) if isinstance(za, (list, tuple)) else za.shape[0]
            elif hasattr(acom.crystal, 'orientations'):
                n_templates = len(acom.crystal.orientations)

            badge_text = f"📚 {cif_name} ({n_templates} za)" if n_templates > 0 else f"📚 {cif_name}"
            btn_cryst_lib.configure(
                text=badge_text,
                fg_color="#1b5e20",
                hover_color="#2e7d32",
                text_color="#76ff03"
            )
        else:
            btn_cryst_lib.configure(
                text="📚 Library: None",
                fg_color="#37474f",
                hover_color="#263238",
                text_color="#cfd8dc"
            )

    def run_orientation_indexing():
        if not state['peaks']:
            messagebox.showwarning("No Peaks", "Fit Bragg peaks before running orientation matching.", parent=root)
            return
        try:
            M_active = np.array([
                [state['eps11'], state['eps12']], 
                [state['eps21'], state['eps22']]
            ], dtype=np.float64)

            candidates = state['acom_mgr'].match_experimental_peaks(
                state['peaks'], 
                num_matches=3, 
                distortion_matrix=M_active
            )
            state['orientation_candidates'] = candidates
            if candidates:
                state['selected_candidate_idx'] = 0
                state['template_rot_offset'] = 0.0
                slider_rot.set(0.0)
                lbl_rot_val.configure(text="Δθ: +0.0°")
                menu_cand_values = [
                    f"#{c['rank']}: [{int(c['zone_axis'][0])} {int(c['zone_axis'][1])} {int(c['zone_axis'][2])}] (Q={c['correlation']:.2f})"
                    for c in candidates
                ]
                cb_cand.configure(values=menu_cand_values)
                var_cand.set(menu_cand_values[0])
                redraw_template_spots()
                lbl_status.configure(text=f"Matched top zone axis: {menu_cand_values[0]}")
            else:
                messagebox.showinfo("No Match", "No orientation matched within correlation threshold.", parent=root)
        except Exception as e:
            messagebox.showerror("Matching Error", str(e), parent=root)

    ctk.CTkButton(f_or_r1, text="🧭 Index", width=65, fg_color="#6a1b9a", hover_color="#4a148c", command=run_orientation_indexing).pack(side=tk.LEFT, padx=3)

    def on_candidate_select(choice):
        for idx, c in enumerate(state['orientation_candidates']):
            if f"#{c['rank']}:" in choice:
                state['selected_candidate_idx'] = idx
                state['template_rot_offset'] = 0.0
                slider_rot.set(0.0)
                lbl_rot_val.configure(text="Δθ: +0.0°")
                redraw_template_spots()
                break

    var_cand = ctk.StringVar(value="Select Zone Axis")
    cb_cand = ctk.CTkOptionMenu(f_or_r1, variable=var_cand, values=["None Indexed"], width=160, font=("Arial", 9), command=on_candidate_select)
    cb_cand.pack(side=tk.LEFT, padx=3)

    f_or_r2 = ctk.CTkFrame(tab_orient, fg_color="transparent")
    f_or_r2.pack(fill=tk.X, pady=(2, 1))

    lbl_rot_val = ctk.CTkLabel(f_or_r2, text="Δθ: +0.0°", width=55, font=("Arial", 10, "bold"), text_color="#76ff03", anchor='w')
    lbl_rot_val.pack(side=tk.LEFT, padx=(2, 2))

    def on_template_rot_slider(val):
        deg = float(val)
        state['template_rot_offset'] = deg
        lbl_rot_val.configure(text=f"Δθ: {deg:+.1f}°")
        redraw_template_spots()

    slider_rot = ctk.CTkSlider(f_or_r2, from_=-15.0, to=15.0, number_of_steps=120, command=on_template_rot_slider, width=170)
    slider_rot.set(0.0)
    slider_rot.pack(side=tk.LEFT, padx=2)

    ctk.CTkButton(f_or_r2, text="↺", width=26, fg_color="#37474f", hover_color="#263238", command=lambda: [slider_rot.set(0.0), on_template_rot_slider(0.0)]).pack(side=tk.LEFT, padx=2)

    var_show_template = tk.BooleanVar(value=state['show_template_overlay'])
    def on_toggle_template_overlay():
        state['show_template_overlay'] = var_show_template.get()
        redraw_template_spots()

    chk_show_template = ctk.CTkSwitch(
        f_or_r2, text="Overlay", variable=var_show_template, 
        command=on_toggle_template_overlay, font=("Arial", 10), width=65
    )
    chk_show_template.pack(side=tk.LEFT, padx=(6, 2))

    f_or_r3 = ctk.CTkFrame(tab_orient, fg_color="transparent")
    f_or_r3.pack(fill=tk.X, pady=(2, 2))

    lbl_ithresh = ctk.CTkLabel(
        f_or_r3, 
        text=f"Min I: {state['template_intensity_thresh']*100:.1f}%", 
        width=70, 
        font=("Arial", 10), 
        anchor='w'
    )
    lbl_ithresh.pack(side=tk.LEFT, padx=(2, 2))

    def on_template_ithresh_slider(val):
        pct = float(val)
        frac = pct / 100.0
        state['template_intensity_thresh'] = frac
        lbl_ithresh.configure(text=f"Min I: {pct:.1f}%")
        redraw_template_spots()

    slider_ithresh = ctk.CTkSlider(
        f_or_r3, 
        from_=0.0, 
        to=20.0, 
        number_of_steps=100, 
        command=on_template_ithresh_slider, 
        width=170
    )
    slider_ithresh.set(state['template_intensity_thresh'] * 100.0)
    slider_ithresh.pack(side=tk.LEFT, padx=2)

    def reset_ithresh():
        slider_ithresh.set(1.0)
        on_template_ithresh_slider(1.0)

    ctk.CTkButton(
        f_or_r3, 
        text="↺", 
        width=26, 
        fg_color="#37474f", 
        hover_color="#263238", 
        command=reset_ithresh
    ).pack(side=tk.LEFT, padx=2)

    update_crystal_badge()

    # TAB 5: Mapping Controls
    tab_mapping = tab_tools.tab("Mapping")

    f_roi_row = ctk.CTkFrame(tab_mapping, fg_color="transparent")
    f_roi_row.pack(fill=tk.X, pady=(2, 2))

    var_use_map_roi = tk.BooleanVar(value=state.get('use_map_roi', False))

    def on_toggle_map_roi():
        state['use_map_roi'] = var_use_map_roi.get()
        if state['use_map_roi']:
            y_min = max(0, state['y'] - state['dy'])
            y_max = min(state['scany'] - 1, state['y'] + state['dy'])
            x_min = max(0, state['x'] - state['dx'])
            x_max = min(state['scanx'] - 1, state['x'] + state['dx'])
            state['map_roi'] = {'xmin': x_min, 'xmax': x_max, 'ymin': y_min, 'ymax': y_max}
        update_map_roi_visual()
        canvas.draw_idle()

    chk_use_roi = ctk.CTkCheckBox(
        f_roi_row, text="Restrict to Subregion ROI", 
        variable=var_use_map_roi, 
        command=on_toggle_map_roi, 
        font=("Arial", 11, "bold")
    )
    chk_use_roi.pack(side=tk.LEFT, padx=3)

    def set_roi_from_current():
        if state['myset'] is None: return
        y_min = max(0, state['y'] - state['dy'])
        y_max = min(state['scany'] - 1, state['y'] + state['dy'])
        x_min = max(0, state['x'] - state['dx'])
        x_max = min(state['scanx'] - 1, state['x'] + state['dx'])
        state['map_roi'] = {'xmin': x_min, 'xmax': x_max, 'ymin': y_min, 'ymax': y_max}
        var_use_map_roi.set(True)
        state['use_map_roi'] = True
        update_map_roi_visual()
        canvas.draw_idle()
        lbl_status.configure(text=f"Mapping ROI set to [{x_min}:{x_max}, {y_min}:{y_max}] px ({(x_max-x_min+1)*(y_max-y_min+1)} pixels).")

    btn_set_roi = ctk.CTkButton(
        f_roi_row, text="📐 Current ROI -> Map Area", width=170, 
        fg_color="#37474f", hover_color="#263238", 
        command=set_roi_from_current
    )
    btn_set_roi.pack(side=tk.LEFT, padx=6)

    f_map_r1 = ctk.CTkFrame(tab_mapping, fg_color="transparent")
    f_map_r1.pack(fill=tk.X, pady=(2, 2))

    ctk.CTkButton(f_map_r1, text="📊 Spot Tensor", width=95, fg_color="#00838f", hover_color="#006064", command=lambda: compute_lattice_strain()).pack(side=tk.LEFT, padx=3)
    ctk.CTkButton(f_map_r1, text="🗺️ Map 2D Strain", width=115, fg_color="#00695c", hover_color="#004d40", command=run_batch_strain_mapping).pack(side=tk.LEFT, padx=3)
    btn_show_map_win = ctk.CTkButton(f_map_r1, text="🪟 View Strain Maps", width=125, fg_color="#37474f", hover_color="#263238", command=open_mapping_window)
    btn_show_map_win.pack(side=tk.LEFT, padx=3)

    f_map_r2 = ctk.CTkFrame(tab_mapping, fg_color="transparent")
    f_map_r2.pack(fill=tk.X, pady=(2, 2))

    ctk.CTkButton(f_map_r2, text="🧭 Map Orientation", width=130, fg_color="#6a1b9a", hover_color="#4a148c", command=run_batch_orientation_mapping).pack(side=tk.LEFT, padx=3)
    btn_show_orient_win = ctk.CTkButton(f_map_r2, text="🪟 View Orientation Maps", width=145, fg_color="#37474f", hover_color="#263238", command=open_orient_mapping_window)
    btn_show_orient_win.pack(side=tk.LEFT, padx=3)


    f_map_r3 = ctk.CTkFrame(tab_mapping, fg_color="transparent")
    f_map_r3.pack(fill=tk.X, pady=(2, 2))

    btn_batch_peaks = ctk.CTkButton(
        f_map_r3,
        text="⚡ Map All Peaks (ROI)",
        width=170,
        fg_color="#00897b",
        hover_color="#00695c",
        command=run_batch_peak_detection
    )
    btn_batch_peaks.pack(side=tk.LEFT, padx=3)

    btn_powder_plot = ctk.CTkButton(
        f_map_r3,
        text="📈 Powder Plot",
        width=130,
        fg_color="#0288d1",
        hover_color="#01579b",
        command=open_powder_plot_dialog
    )
    btn_powder_plot.pack(side=tk.LEFT, padx=3)

    def open_selective_map_dialog():
        if not state.get('framepeaklist'):
            messagebox.showwarning("No Peaks", "Run 'Map All Peaks (ROI)' first.", parent=root)
            return

        # Default to a central 25%-75% q-interval if unopened
        all_q = [p['q'] for p in state['framepeaklist'] if len(p.get('q', [])) > 0]
        if all_q:
            flat_q = np.concatenate(all_q)
            q_min = float(np.percentile(flat_q, 30))
            q_max = float(np.percentile(flat_q, 70))
        else:
            q_min, q_max = 1.0, 3.0

        on_powder_range_selected(q_min, q_max)

    btn_sel_map = ctk.CTkButton(
        f_map_r3,
        text="🗺️ Selective Map",
        width=130,
        fg_color="#6a1b9a",
        hover_color="#4a148c",
        command=open_selective_map_dialog
    )
    btn_sel_map.pack(side=tk.LEFT, padx=3)

    # TAB 6: Export
    tab_exp = tab_tools.tab("Export")
    var_export_markers = tk.BooleanVar(value=config.get("export_markers", True))

    f_exp_fig = ctk.CTkFrame(tab_exp, fg_color="transparent")
    f_exp_fig.pack(fill=tk.X, pady=(4, 4))

    chk_exp_markers = ctk.CTkCheckBox(
        f_exp_fig, text="Include Markers & Overlays", variable=var_export_markers, 
        command=lambda: [config.__setitem__('export_markers', var_export_markers.get()), save_config(config)], 
        font=("Arial", 11), width=180
    )
    chk_exp_markers.pack(side=tk.LEFT, padx=4)

    def save_figure_callback():
        if state['myset'] is None:
            messagebox.showwarning("No Data", "Please load a dataset before saving.")
            return

        next_idx, filename = get_next_filename()
        include_markers = var_export_markers.get()
        overlay_artists = [
            artists.get('v_line'), artists.get('h_line'), artists.get('roi_rect'),
            artists.get('center_marker'), artists.get('circle_in'), artists.get('circle_out'),
            artists.get('ruler1_line'), artists.get('ruler2_line'),
            artists.get('map_roi_rect')
        ]
        if ruler1_state['active']: overlay_artists.append(artists.get('ruler1_text'))
        if ruler2_state['active']: overlay_artists.append(artists.get('ruler2_text'))
        if state['show_stats']:
            overlay_artists.append(artists.get('stats_box_left'))
            overlay_artists.append(artists.get('stats_box_right'))
        overlay_artists.extend(artists.get('ring_patches', []))
        overlay_artists.extend(artists.get('calib_patches', []))
        overlay_artists.extend(artists.get('peak_markers', []))
        overlay_artists.extend(artists.get('template_patches', []))
        overlay_artists.extend(artists.get('strain_lines', []))
        overlay_artists = [a for a in overlay_artists if a is not None]

        saved_vis = [art.get_visible() for art in overlay_artists]
        try:
            if not include_markers:
                for art in overlay_artists: art.set_visible(False)
                canvas.draw()
            fig.savefig(filename, bbox_inches='tight', pad_inches=0.05)
        except Exception as e:
            messagebox.showerror("Export Error", str(e))
            return
        finally:
            for art, vis in zip(overlay_artists, saved_vis): art.set_visible(vis)
            canvas.draw_idle()

        next_idx_new, _ = get_next_filename()
        btn_save.configure(text=f"Export Fig #{next_idx_new:03d}")
        messagebox.showinfo("Export Success", f"Saved:\n{filename}")

    btn_save = ctk.CTkButton(f_exp_fig, text="Export Fig #001", width=140, command=save_figure_callback)
    btn_save.pack(side=tk.LEFT, padx=8)

    def on_press(event):
        if state['myset'] is None or event.inaxes is None: return
        if event.button == 3 and event.inaxes == ax2:
            rebuild_context_menu()
            ctx_menu.tk_popup(event.guiEvent.x_root, event.guiEvent.y_root)
            return

        if event.button == 1:
            curr_time = time.time()
            dt = curr_time - click_tracker['last_time']
            is_double_click = (dt < 0.35) and (click_tracker['last_ax'] == event.inaxes)
            is_shift_click = (event.key == 'shift')
            click_tracker['last_time'] = curr_time
            click_tracker['last_ax'] = event.inaxes

            if is_shift_click:
                if event.inaxes == ax1 and ruler1_state['active']:
                    pt = (event.xdata, event.ydata)
                    if ruler1_state['p1'] is None or ruler1_state['p2'] is not None:
                        ruler1_state['p1'], ruler1_state['p2'] = pt, None
                        artists['ruler1_line'].set_data([pt[0]], [pt[1]])
                        artists['ruler1_text'].set_text("Shift+Click 2nd point")
                        artists['ruler1_text'].set_visible(True)
                    else:
                        ruler1_state['p2'] = pt
                        p1 = ruler1_state['p1']
                        artists['ruler1_line'].set_data([p1[0], pt[0]], [p1[1], pt[1]])
                        dr = np.hypot(pt[0] - p1[0], pt[1] - p1[1])
                        ruler1_state['length'] = dr
                        sync_target_dict()
                        artists['ruler1_text'].set_text(f"Δr = {dr:.3f} nm\n({dr/state['re_samp']:.1f} px)")
                        artists['ruler1_text'].set_visible(True)
                    canvas.draw_idle()
                    return

                elif event.inaxes == ax2 and ruler2_state['active']:
                    pt = (event.xdata, event.ydata)
                    if ruler2_state['p1'] is None or ruler2_state['p2'] is not None:
                        ruler2_state['p1'], ruler2_state['p2'] = pt, None
                        artists['ruler2_line'].set_data([pt[0]], [pt[1]])
                        artists['ruler2_text'].set_text("Shift+Click 2nd point")
                        artists['ruler2_text'].set_visible(True)
                    else:
                        ruler2_state['p2'] = pt
                        p1 = ruler2_state['p1']
                        artists['ruler2_line'].set_data([p1[0], pt[0]], [p1[1], pt[1]])
                        raw_dq = np.array([pt[0] - p1[0], pt[1] - p1[1]], dtype=float)
                        M = np.array([[state['eps11'], state['eps12']], [state['eps21'], state['eps22']]], dtype=float)
                        dq = float(np.linalg.norm(M @ raw_dq))
                        d_spacing = 1.0 / dq if dq > 1e-6 else np.inf
                        ruler2_state['dq'], ruler2_state['d_spacing'] = dq, d_spacing
                        sync_target_dict()
                        artists['ruler2_text'].set_text(f"Δq = {dq:.4f} 1/nm\nd = {d_spacing:.4f} nm")
                        artists['ruler2_text'].set_visible(True)
                    canvas.draw_idle()
                    return

            pan_state['active'] = True
            pan_state['ax'] = event.inaxes
            pan_state['start_x'], pan_state['start_y'] = event.x, event.y
            pan_state['xlim'], pan_state['ylim'] = event.inaxes.get_xlim(), event.inaxes.get_ylim()

            if is_double_click:
                if event.inaxes == ax1:
                    state['x'] = int(np.clip(event.xdata / state['re_samp'], 0, state['scanx'] - 1))
                    state['y'] = int(np.clip(state['scany'] - 1 - (event.ydata / state['re_samp']) if state['left_origin'] == 'upper' else event.ydata / state['re_samp'], 0, state['scany'] - 1))
                    px_x = state['x'] * state['re_samp']
                    px_y = (state['scany'] - 1 - state['y']) * state['re_samp'] if state['left_origin'] == 'upper' else state['y'] * state['re_samp']
                    artists['v_line'].set_xdata([px_x, px_x])
                    artists['h_line'].set_ydata([px_y, px_y])
                    refresh_spatial_roi()
                elif event.inaxes == ax2:
                    delta_qx_px = event.xdata / state['rec_samp']
                    delta_qy_px = event.ydata / state['rec_samp']
                    state['cqx'] = float(np.clip(state['cqx'] + delta_qx_px, 0, state['detx'] - 1))
                    state['cqy'] = float(np.clip(state['cqy'] - delta_qy_px if state['right_origin'] == 'upper' else state['cqy'] + delta_qy_px, 0, state['dety'] - 1))
                    refresh_detector_mask()

    def on_release(event):
        pan_state['active'] = False
        pan_state['ax'] = None

    def on_motion(event):
        if state['myset'] is None: return

        if event.inaxes == ax1 and event.xdata is not None and event.ydata is not None:
            phys_x, phys_y = event.xdata, event.ydata
            px_x = int(np.clip(phys_x / state['re_samp'], 0, state['scanx'] - 1))
            px_y = int(np.clip(state['scany'] - 1 - (phys_y / state['re_samp']) if state['left_origin'] == 'upper' else phys_y / state['re_samp'], 0, state['scany'] - 1))
            
            if 'data_left_raw' in state and state['data_left_raw'] is not None:
                val = state['data_left_raw'][px_y, px_x]
                lbl_status.configure(text=f"Real View: [x: {phys_x:.2f} nm, y: {phys_y:.2f} nm] | Pixel: ({px_y}, {px_x}) | Value: {val:.3e}")

        elif event.inaxes == ax2 and event.xdata is not None and event.ydata is not None:
            qx, qy = event.xdata, event.ydata
            det_x = int(np.clip(qx / state['rec_samp'] + state['cqx'], 0, state['detx'] - 1))
            det_y = int(np.clip(state['cqy'] - (qy / state['rec_samp']) if state['right_origin'] == 'upper' else qy / state['rec_samp'] + state['cqy'], 0, state['dety'] - 1))
            intensity = state['norm_img'][det_y, det_x] if state['norm_img'] is not None else 0.0
            d_spacing = 1.0 / np.hypot(qx, qy) if np.hypot(qx, qy) > 1e-6 else np.inf
            lbl_status.configure(text=f"Recip View: [qx: {qx:.3f}, qy: {qy:.3f} 1/nm] (d: {d_spacing:.3f} nm) | Det: ({det_y}, {det_x}) | I: {intensity:.3e}")

        if pan_state['active'] and pan_state['ax'] == event.inaxes and event.x is not None and event.y is not None:
            ax = pan_state['ax']
            bbox = ax.get_window_extent()
            dx_data = -(event.x - pan_state['start_x']) * (pan_state['xlim'][1] - pan_state['xlim'][0]) / bbox.width
            dy_data = -(event.y - pan_state['start_y']) * (pan_state['ylim'][1] - pan_state['ylim'][0]) / bbox.height
            ax.set_xlim(pan_state['xlim'][0] + dx_data, pan_state['xlim'][1] + dx_data)
            ax.set_ylim(pan_state['ylim'][0] + dy_data, pan_state['ylim'][1] + dy_data)
            canvas.draw_idle()

    def on_scroll(event):
        if state['myset'] is None or event.inaxes is None: return
        ax = event.inaxes
        scale_factor = 0.85 if event.button == 'up' else 1.15
        xdata, ydata = event.xdata, event.ydata
        if xdata is None or ydata is None: return

        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        new_w, new_h = (xlim[1] - xlim[0]) * scale_factor, (ylim[1] - ylim[0]) * scale_factor
        rel_x, rel_y = (xdata - xlim[0]) / (xlim[1] - xlim[0]), (ydata - ylim[0]) / (ylim[1] - ylim[0])
        ax.set_xlim([xdata - rel_x * new_w, xdata + (1.0 - rel_x) * new_w])
        ax.set_ylim([ydata - rel_y * new_h, ydata + (1.0 - rel_y) * new_h])
        canvas.draw_idle()

    canvas.mpl_connect('button_press_event', on_press)
    canvas.mpl_connect('button_release_event', on_release)
    canvas.mpl_connect('motion_notify_event', on_motion)
    canvas.mpl_connect('scroll_event', on_scroll)

    rebuild_plot()
    root.update_idletasks()
    canvas.draw_idle()

    root.mainloop()
    return target_dict


def main():
    VirtualSTEMExplorerCTK(initial_filepath=None)

if __name__ == "__main__":
    main()

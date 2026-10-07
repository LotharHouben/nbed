import numpy as np
import numpy.matlib
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display
from matplotlib.widgets import Slider, RangeSlider, TextBox, Button, RadioButtons
from matplotlib.patches import Rectangle, Circle
import glob
import re

def select_frame(myset, vimage, target_dict=None, initial_coords=None, initial_dx=0, initial_dy=0):
    """
    Interactive selection tool that updates target_dict in real-time,
    with dx and dy ROI spatial averaging over myset.data.
    """
    global event_data
    if target_dict is None:
        target_dict = event_data

    scany, scanx = myset.dim[0], myset.dim[1]
    
    if initial_coords is None:
        y, x = scany // 2, scanx // 2
    else:
        y, x = initial_coords

    dx, dy = int(initial_dx), int(initial_dy)

    # Available colormaps to choose from
    cmap_options = ['viridis', 'viridis_r', 'gray', 'gray_r', 'inferno', 'turbo', 'cividis']
    current_cmap = {'name': 'viridis'}
    
    # Initialize shared dictionary state
    target_dict['x'] = x
    target_dict['y'] = y
    target_dict['dx'] = dx
    target_dict['dy'] = dy
    target_dict['index'] = y * scanx + x
    target_dict['cmap'] = current_cmap['name']
    
    data_left_raw = vimage

    # Helper for averaged ROI extraction
    def extract_avg_frame(center_y, center_x, radius_y, radius_x):
        y_min = max(0, center_y - radius_y)
        y_max = min(scany, center_y + radius_y + 1)
        x_min = max(0, center_x - radius_x)
        x_max = min(scanx, center_x + radius_x + 1)
        
        # Mean across spatial slice (y, x)
        raw_roi = myset.data[y_min:y_max, x_min:x_max, :, :]
        avg_slice = np.mean(raw_roi, axis=(0, 1))
        return np.log(avg_slice + 1.0)

    data_right_raw = extract_avg_frame(y, x, dy, dx)

    def apply_contrast(data, clip_percentiles, gamma):
        p_low, p_high = clip_percentiles
        vmin, vmax = np.percentile(data, [p_low, p_high])
        if vmax == vmin:
            vmax = vmin + 1e-5
        norm = np.clip((data - vmin) / (vmax - vmin), 0.0, 1.0)
        return np.power(norm, gamma)

    # --- Figure Setup ---
    fig = plt.figure(figsize=(11, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[3.0, 1.2], hspace=0.38)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    img1_proc = apply_contrast(data_left_raw, (0, 100), 1.0)
    img2_proc = apply_contrast(data_right_raw, (0, 100), 1.0)

    im1 = ax1.imshow(img1_proc, cmap='gray', origin="lower", vmin=0, vmax=1)
    ax1.set_title("Virtual Image")
    v_line = ax1.axvline(x, color='yellow', lw=1)
    h_line = ax1.axhline(y, color='yellow', lw=1)

    # ROI Rectangle Box indicator on left image
    roi_rect = Rectangle((x - dx - 0.5, y - dy - 0.5), 2*dx + 1, 2*dy + 1,
                         linewidth=1, edgecolor='yellow', facecolor='none', linestyle='-')
    ax1.add_patch(roi_rect)

    im2 = ax2.imshow(img2_proc, cmap=current_cmap['name'], origin="lower", vmin=0, vmax=1)
    ax2.set_title(f"Frame: ({y}, {x}), dx:{dx}, dy:{dy}, Index: {target_dict['index']}")

    # --- Widget Controls & Layout ---
    # Left Controls Axes (Contrast + Reset)
    ax_left_clip     = fig.add_axes([0.15, 0.3, 0.15, 0.025])
    ax_left_low_tb   = fig.add_axes([0.39, 0.3, 0.05, 0.025])
    ax_left_high_tb  = fig.add_axes([0.44, 0.3, 0.05, 0.025])
    
    ax_left_gamma    = fig.add_axes([0.15, 0.27, 0.15, 0.025])
    ax_left_gamma_tb = fig.add_axes([0.39, 0.27, 0.05, 0.025])
    ax_left_reset    = fig.add_axes([0.39, 0.22, 0.08, 0.025])

    # Right Controls Axes (Contrast + Reset)
    ax_right_clip     = fig.add_axes([0.60, 0.3, 0.15, 0.025])
    ax_right_low_tb   = fig.add_axes([0.84, 0.3, 0.05, 0.025])
    ax_right_high_tb  = fig.add_axes([0.89, 0.3, 0.05, 0.025])
    
    ax_right_gamma    = fig.add_axes([0.60, 0.27, 0.15, 0.025])
    ax_right_gamma_tb = fig.add_axes([0.84, 0.27, 0.04, 0.025])
    ax_right_reset    = fig.add_axes([0.84, 0.22, 0.08, 0.025])

    # Spatial ROI dx / dy Axes (Top Center/Right)
    ax_dx_tb = fig.add_axes([0.20, 0.18, 0.05, 0.025])
    ax_dy_tb = fig.add_axes([0.25, 0.18, 0.05, 0.025])

    # Colormap Radio Selector Axis (Positioned to the right of controls)
    ax_cmap = fig.add_axes([0.6, 0.09, 0.15, 0.16])
    ax_cmap.set_title("", fontsize=8)
    
    # Instantiate Sliders & Buttons
    s_left_clip = RangeSlider(ax_left_clip, 'Clip % ', 0.0, 100.0, valinit=(0, 100))
    s_left_gamma = Slider(ax_left_gamma,    'gamma  ', 0.1, 3.0, valinit=1.0)
    btn_left_reset = Button(ax_left_reset, ' Reset ')

    s_right_clip = RangeSlider(ax_right_clip, 'Clip % ', 0.0, 100.0, valinit=(0, 100))
    s_right_gamma = Slider(ax_right_gamma,    'gamma  ', 0.1, 3.0, valinit=1.0)
    btn_right_reset = Button(ax_right_reset, ' Reset ')

    radio_cmap = RadioButtons(ax_cmap, cmap_options, active=0)

    # Instantiate TextBoxes
    tb_left_low   = TextBox(ax_left_low_tb, '', initial='0.0')
    tb_left_high  = TextBox(ax_left_high_tb, '', initial='100')
    tb_left_gamma = TextBox(ax_left_gamma_tb, '', initial='1.0')

    tb_right_low   = TextBox(ax_right_low_tb, '', initial='0.0')
    tb_right_high  = TextBox(ax_right_high_tb, '', initial='100')
    tb_right_gamma = TextBox(ax_right_gamma_tb, '', initial='1.0')

    tb_dx = TextBox(ax_dx_tb, ' Region dx, dy: ', initial=str(dx))
    tb_dy = TextBox(ax_dy_tb, '', initial=str(dy))

    updating = False

    def update_left(_=None):
        nonlocal updating
        processed = apply_contrast(data_left_raw, s_left_clip.val, s_left_gamma.val)
        im1.set_data(processed)
        if not updating:
            updating = True
            tb_left_low.set_val(f"{s_left_clip.val[0]:.1f}")
            tb_left_high.set_val(f"{s_left_clip.val[1]:.1f}")
            tb_left_gamma.set_val(f"{s_left_gamma.val:.2f}")
            updating = False
        fig.canvas.draw_idle()

    def update_right(_=None):
        nonlocal updating
        processed = apply_contrast(data_right_raw, s_right_clip.val, s_right_gamma.val)
        im2.set_data(processed)
        if not updating:
            updating = True
            tb_right_low.set_val(f"{s_right_clip.val[0]:.1f}")
            tb_right_high.set_val(f"{s_right_clip.val[1]:.1f}")
            tb_right_gamma.set_val(f"{s_right_gamma.val:.2f}")
            updating = False
        fig.canvas.draw_idle()

    def change_cmap(label):
        current_cmap['name'] = label
        target_dict['cmap'] = label
        im2.set_cmap(label)
        fig.canvas.draw_idle()
        
    def refresh_roi(_=None):
        nonlocal data_right_raw, dx, dy
        try:
            dx = max(0, int(tb_dx.text))
            dy = max(0, int(tb_dy.text))
        except ValueError:
            return

        target_dict['dx'] = dx
        target_dict['dy'] = dy

        # Update ROI rectangle on ax1
        roi_rect.set_bounds(target_dict['x'] - dx - 0.5, target_dict['y'] - dy - 0.5, 2*dx + 1, 2*dy + 1)
        
        # Re-extract and refresh frame
        data_right_raw = extract_avg_frame(target_dict['y'], target_dict['x'], dy, dx)
        ax2.set_title(f"Frame: ({target_dict['y']}, {target_dict['x']}), dx:{dx}, dy:{dy}, Index: {target_dict['index']}")
        update_right()

    def reset_left(_=None):
        nonlocal updating
        updating = True
        s_left_clip.set_val((0.0, 100.0)); s_left_gamma.set_val(1.0)
        tb_left_low.set_val('0.0'); tb_left_high.set_val('100.0'); tb_left_gamma.set_val('1.0')
        updating = False
        update_left()

    def reset_right(_=None):
        nonlocal updating
        updating = True
        s_right_clip.set_val((0.0, 100.0)); s_right_gamma.set_val(1.0)
        tb_right_low.set_val('0.0'); tb_right_high.set_val('100.0'); tb_right_gamma.set_val('1.0')
        updating = False
        update_right()

    def submit_left_clip(_=None):
        nonlocal updating
        if updating: return
        try:
            updating = True
            s_left_clip.set_val((float(tb_left_low.text), float(tb_left_high.text)))
            updating = False
            update_left()
        except ValueError: pass

    def submit_left_gamma(_=None):
        nonlocal updating
        if updating: return
        try:
            updating = True
            s_left_gamma.set_val(float(tb_left_gamma.text))
            updating = False
            update_left()
        except ValueError: pass

    def submit_right_clip(_=None):
        nonlocal updating
        if updating: return
        try:
            updating = True
            s_right_clip.set_val((float(tb_right_low.text), float(tb_right_high.text)))
            updating = False
            update_right()
        except ValueError: pass

    def submit_right_gamma(_=None):
        nonlocal updating
        if updating: return
        try:
            updating = True
            s_right_gamma.set_val(float(tb_right_gamma.text))
            updating = False
            update_right()
        except ValueError: pass

    # Event Bindings
    s_left_clip.on_changed(update_left); s_left_gamma.on_changed(update_left)
    s_right_clip.on_changed(update_right); s_right_gamma.on_changed(update_right)
    btn_left_reset.on_clicked(reset_left); btn_right_reset.on_clicked(reset_right)
    
    tb_left_low.on_submit(submit_left_clip); tb_left_high.on_submit(submit_left_clip); tb_left_gamma.on_submit(submit_left_gamma)
    tb_right_low.on_submit(submit_right_clip); tb_right_high.on_submit(submit_right_clip); tb_right_gamma.on_submit(submit_right_gamma)
    radio_cmap.on_clicked(change_cmap)
    
    tb_dx.on_submit(refresh_roi)
    tb_dy.on_submit(refresh_roi)

    def onclick(event):
        if event.inaxes == ax1:
            click_x, click_y = int(event.xdata), int(event.ydata)
            
            v_line.set_xdata([click_x, click_x])
            h_line.set_ydata([click_y, click_y])
            
            target_dict['x'] = click_x
            target_dict['y'] = click_y
            target_dict['index'] = click_y * scanx + click_x
            
            refresh_roi()

    fig.canvas.mpl_connect('button_press_event', onclick)

    fig._widgets = [
        tb_dx, tb_dy, 
        s_left_clip, s_left_gamma, s_right_clip, s_right_gamma,
        tb_left_low, tb_left_high, tb_left_gamma,
        tb_right_low, tb_right_high, tb_right_gamma,
        btn_left_reset, btn_right_reset, radio_cmap
    ]

    plt.show()
    return target_dict



def VirtualSTEM_Explorer(
    myset, 
    target_dict=None, 
    initial_coords=None, 
    initial_dx=5, 
    initial_dy=5,
    re_samp=1.0,      # Real-space sampling (e.g., nm / pixel)
    re_unit="nm",
    rec_samp=1.0,     # Reciprocal-space sampling (e.g., 1/nm / pixel)
    rec_unit="1/nm",
    save_prefix="VirtualSTEMExplorer_Figure",  # Default save path
    save_fmt='pdf' # 'pdf', 'png', 'svg', etc.
        
):
    """
    Interactive 4D-STEM Virtual Explorer with Dual Axes, Calibrated Measurement Rulers, and Continuous State Sync.

    This interactive Jupyter/ipympl widget provides real-time exploration of 4D-STEM datacubes.
    It integrates virtual image reconstruction across multiple contrast/integration modes (Sum,
    Variance, Fluctuation, Center of Mass) with spatially averaged virtual diffraction pattern 
    inspection, dual physical/pixel coordinate axes, interactive measurement rulers, and high-resolution 
    publication figure exporting.

    Parameters
    ----------
    myset : object
        4D-STEM dataset container or dataset object. Must expose a `.dim` attribute with shape 
        `(scany, scanx, dety, detx)` and a 4D NumPy array `.data` supporting slicing as 
        `data[y_min:y_max, x_min:x_max, :, :]`.
    target_dict : dict, optional
        Dictionary used to continuously store active explorer state variables (coordinates, mask settings, 
        colormaps, and ruler outputs). If None, defaults to global `event_data` if defined, or initializes 
        a new dictionary.
    initial_coords : tuple of int, optional
        Starting spatial scan pixel coordinates `(y, x)`. Defaults to center of scan grid `(scany // 2, scanx // 2)`.
    initial_dx : int, optional
        Initial half-width (radius) of spatial region-of-interest (ROI) box in X. Default is 5 pixels.
    initial_dy : int, optional
        Initial half-height (radius) of spatial region-of-interest (ROI) box in Y. Default is 5 pixels.
    re_samp : float, optional
        Real-space pixel sampling pitch (e.g., nanometers per pixel). Default is 1.0.
    re_unit : str, optional
        Unit label string for real-space dimensions (e.g., "nm", "Å", "µm"). Default is "nm".
    rec_samp : float, optional
        Reciprocal-space pixel sampling pitch (e.g., inverse nanometers per pixel). Default is 1.0.
    rec_unit : str, optional
        Unit label string for reciprocal-space dimensions (e.g., "1/nm", "1/Å"). Default is "1/nm".
    save_prefix : str, optional
        Base filename prefix for auto-incrementing figure exports. Default is "VirtualSTEM_Fig".
    save_fmt : str, optional
        File format extension for publication figure exports (e.g., "pdf", "png", "svg", "tiff"). 
        Default is "pdf".

    Returns
    -------
    target_dict : dict
        Reference to the live state container dictionary. Continuously updated during interaction with 
        the following keys:
        - `'x'`, `'y'` : Current spatial probe location in pixel units.
        - `'x_phys'`, `'y_phys'` : Current spatial probe location in physical units (`re_unit`).
        - `'dx'`, `'dy'` : Spatial ROI half-widths in pixels.
        - `'cqx'`, `'cqy'` : Reciprocal detector central beam center in pixels.
        - `'r_in'`, `'r_out'` : Detector annular mask inner and outer radii in pixels.
        - `'r_in_phys'`, `'r_out_phys'` : Detector annular mask radii in physical units (`rec_unit`).
        - `'left_cmap'`, `'right_cmap'` : Active colormap strings for left and right panels.
        - `'mode'` : Selected virtual integration mode ('Sum', 'Variance', 'Fluctuation', 'COM Mag', 'COM Azimuth').
        - `'index'` : Flattened 1D spatial frame index (`y * scanx + x`).
        - `'ruler_real_dr'` : Active measurement length on real-space virtual image in physical units (`re_unit`).
        - `'ruler_recip_dq'` : Active measurement vector magnitude on diffraction pattern in physical units (`rec_unit`).
        - `'ruler_recip_d'` : Calculated interplanar lattice spacing `d = 1/dq` in real-space physical units (`re_unit`).

    Features
    --------
    - Dual Panel Axes: Displays calibrated physical units on primary bottom/left axes alongside 
      pixel coordinate indices on secondary top/right axes.
    - Real & Reciprocal Rulers:
      - 'Ruler Real': Measures real-space distances (Delta r) and pixel counts across virtual STEM reconstructions.
      - 'Ruler Recip': Measures scattering vectors (Delta q) and computes corresponding real-space $d$-spacings.
    - State Synchronization: Automatically updates `target_dict` on any mouse click, slider change, text box submission, 
      or mode change.
    - Publication Export: Includes a 'Save Fig' button that runs Matplotlib's native `fig.savefig()`, auto-incrementing 
      filenames (e.g., `VirtualSTEM_Fig_001.pdf`) while respecting global `rcParams` settings.

    Notes
    -----
    Requires `%matplotlib widget` backend (`ipympl`) to be enabled prior to function execution inside Jupyter notebooks.
    """
    global event_data
    if target_dict is None:
        try:
            target_dict = event_data
        except NameError:
            target_dict = {}
            
    # Auto-increment file counter tracker
    def get_next_filename():
        """Scans directory for existing prefix_XXX.fmt files and returns (next_idx, filename)."""
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

    # Initialize first file counter state
    init_idx, _ = get_next_filename()
            
    scany, scanx = myset.dim[0], myset.dim[1]
    dety, detx = myset.dim[2], myset.dim[3]
    
    # Coordinates in pixel units
    if initial_coords is None:
        y, x = scany // 2, scanx // 2
    else:
        y, x = initial_coords
    dx, dy = int(initial_dx), int(initial_dy)

    # Reciprocal Detector Coordinates in pixels
    cqy, cqx = dety // 2, detx // 2
    r_in, r_out = 0.0, float(min(dety, detx) // 4)

    qy_grid, qx_grid = np.ogrid[:dety, :detx]

    # Extents for Calibrated Physical Display
    extent_left = [0, scanx * re_samp, 0, scany * re_samp]
    
    def get_reciprocal_extent(center_x, center_y):
        x_min = -center_x * rec_samp
        x_max = (detx - center_x) * rec_samp
        y_min = -center_y * rec_samp
        y_max = (dety - center_y) * rec_samp
        return [x_min, x_max, y_min, y_max]

    extent_right = get_reciprocal_extent(cqx, cqy)

    cmap_options_left = ['gray', 'gray_r', 'inferno', 'magma', 'viridis', 'plasma', 'turbo', 'cividis', 'hsv', 'twilight', 'twilight_shifted']
    cmap_options_right = ['gray', 'gray_r', 'inferno', 'magma', 'viridis', 'plasma', 'turbo', 'cividis']
    
    left_cmap = {'name': 'gray'}
    right_cmap = {'name': 'turbo'}
    calc_mode = {'mode': 'Sum'}

    # Ruler measurement containers
    ruler1_state = {'active': False, 'p1': None, 'p2': None, 'length': 0.0}
    ruler2_state = {'active': False, 'p1': None, 'p2': None, 'dq': 0.0, 'd_spacing': 0.0}

    # Central State Sync Function
    def sync_target_dict():
        target_dict['x'] = int(x)
        target_dict['y'] = int(y)
        target_dict['x_phys'] = float(x * re_samp)
        target_dict['y_phys'] = float(y * re_samp)
        target_dict['dx'] = int(dx)
        target_dict['dy'] = int(dy)
        target_dict['cqx'] = float(cqx)
        target_dict['cqy'] = float(cqy)
        target_dict['r_in'] = float(r_in)
        target_dict['r_out'] = float(r_out)
        target_dict['r_in_phys'] = float(r_in * rec_samp)
        target_dict['r_out_phys'] = float(r_out * rec_samp)
        target_dict['left_cmap'] = left_cmap['name']
        target_dict['right_cmap'] = right_cmap['name']
        target_dict['mode'] = calc_mode['mode']
        target_dict['index'] = int(y * scanx + x)
        target_dict['ruler_real_dr'] = float(ruler1_state['length'])
        target_dict['ruler_recip_dq'] = float(ruler2_state['dq'])
        target_dict['ruler_recip_d'] = float(ruler2_state['d_spacing'])

    # --- Processing Functions ---
    def compute_virtual_image(center_qx, center_qy, radius_in, radius_out, mode='Sum'):
        dist_sq = (qx_grid - center_qx)**2 + (qy_grid - center_qy)**2
        mask = (dist_sq >= radius_in**2) & (dist_sq <= radius_out**2)
        
        if np.any(mask):
            masked_data = myset.data[:, :, mask]
            if mode == 'Sum':
                vimage = np.sum(masked_data, axis=-1)
            elif mode == 'Variance':
                vimage = np.var(masked_data, axis=-1)
            elif mode == 'Fluctuation':
                mean_val = np.mean(masked_data, axis=-1)
                var_val = np.var(masked_data, axis=-1)
                vimage = np.where(mean_val > 1e-6, var_val / (mean_val**2 + 1e-6), 0.0)
            elif mode in ['COM Mag', 'COM Azimuth']:
                rel_qx = (qx_grid - center_qx)[mask]
                rel_qy = (qy_grid - center_qy)[mask]
                I_total = np.sum(masked_data, axis=-1)
                I_total_safe = np.where(I_total > 1e-6, I_total, 1e-6)
                com_x = np.sum(masked_data * rel_qx, axis=-1) / I_total_safe
                com_y = np.sum(masked_data * rel_qy, axis=-1) / I_total_safe
                if mode == 'COM Mag':
                    vimage = np.hypot(com_x, com_y) * rec_samp
                else:
                    vimage = np.arctan2(com_y, com_x)
            else:
                vimage = np.sum(masked_data, axis=-1)
        else:
            vimage = np.zeros((scany, scanx))
        return vimage, mask

    def extract_avg_diffraction(center_y, center_x, radius_y, radius_x):
        y_min = max(0, center_y - radius_y)
        y_max = min(scany, center_y + radius_y + 1)
        x_min = max(0, center_x - radius_x)
        x_max = min(scanx, center_x + radius_x + 1)
        avg_slice = np.mean(myset.data[y_min:y_max, x_min:x_max, :, :], axis=(0, 1))
        return np.log(avg_slice + 1.0)

    data_left_raw, current_mask = compute_virtual_image(cqx, cqy, r_in, r_out, mode=calc_mode['mode'])
    data_right_raw = extract_avg_diffraction(y, x, dy, dx)

    def apply_contrast(data, clip_percentiles, gamma):
        p_low, p_high = clip_percentiles
        vmin, vmax = np.percentile(data, [p_low, p_high])
        if vmax == vmin:
            vmax = vmin + 1e-5
        norm = np.clip((data - vmin) / (vmax - vmin), 0.0, 1.0)
        return np.power(norm, gamma)

    # --- Layout Setup ---
    plt.ioff()
    fig = plt.figure(figsize=(14.0, 8.5))
    if hasattr(fig.canvas, 'header_visible'):
        fig.canvas.header_visible = False
    fig.subplots_adjust(top=0.86, bottom=0.25, left=0.06, right=0.94)

    gs = fig.add_gridspec(1, 2, wspace=0.35)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])

    img1_proc = apply_contrast(data_left_raw, (0, 100), 1.0)
    img2_proc = apply_contrast(data_right_raw, (0, 100), 1.0)

    # LEFT Panel: Calibrated Primary Axes
    im1 = ax1.imshow(img1_proc, cmap=left_cmap['name'], origin="lower", vmin=0, vmax=1, extent=extent_left)
    ax1.set_title(f"Virtual Image [{calc_mode['mode']}]", pad=25)
    ax1.set_xlabel(f"x ({re_unit})"); ax1.set_ylabel(f"y ({re_unit})")

    # LEFT Panel: Pixel Secondary Axes (Top / Right)
    sec_ax1_x = ax1.secondary_xaxis('top', functions=(lambda val: val / re_samp, lambda val: val * re_samp))
    sec_ax1_y = ax1.secondary_yaxis('right', functions=(lambda val: val / re_samp, lambda val: val * re_samp))
    sec_ax1_x.set_xlabel("x (pixels)", labelpad=6)
    sec_ax1_y.set_ylabel("y (pixels)", labelpad=6)

    px_x, px_y = x * re_samp, y * re_samp
    v_line = ax1.axvline(px_x, color='yellow', lw=1)
    h_line = ax1.axhline(px_y, color='yellow', lw=1)
    roi_rect = Rectangle(((x - dx - 0.5) * re_samp, (y - dy - 0.5) * re_samp), 
                         (2*dx + 1) * re_samp, (2*dy + 1) * re_samp,
                         linewidth=1, edgecolor='yellow', facecolor='none', linestyle='--')
    ax1.add_patch(roi_rect)

    # RIGHT Panel: Calibrated Primary Axes
    im2 = ax2.imshow(img2_proc, cmap=right_cmap['name'], origin="lower", vmin=0, vmax=1, extent=extent_right)
    ax2.set_title(f"Diffraction Frame ({y}, {x})", pad=25)
    ax2.set_xlabel(f"q_x ({rec_unit})"); ax2.set_ylabel(f"q_y ({rec_unit})")

    # RIGHT Panel: Detector Pixel Secondary Axes (Top / Right)
    sec_ax2_x = ax2.secondary_xaxis('top', functions=(lambda val: val / rec_samp + cqx, lambda val: (val - cqx) * rec_samp))
    sec_ax2_y = ax2.secondary_yaxis('right', functions=(lambda val: val / rec_samp + cqy, lambda val: (val - cqy) * rec_samp))
    sec_ax2_x.set_xlabel("q_x (pixels)", labelpad=6)
    sec_ax2_y.set_ylabel("q_y (pixels)", labelpad=6)

    center_marker = ax2.plot(0, 0, 'rx', markersize=8)[0]
    circle_in = Circle((0, 0), r_in * rec_samp, color='yellow', fill=False, linestyle='--', linewidth=1.5)
    circle_out = Circle((0, 0), r_out * rec_samp, color='yellow', fill=False, linestyle='-', linewidth=1.5)
    ax2.add_patch(circle_in)
    ax2.add_patch(circle_out)

    # --- Interactive Ruler Elements ---
    ruler1_line = ax1.plot([], [], 'm--', lw=1.5, marker='o', markersize=4)[0]
    ruler1_text = ax1.text(0.03, 0.95, "", transform=ax1.transAxes, color='magenta', 
                           fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="black", ec="magenta", alpha=0.8))
    ruler1_text.set_visible(False)

    ruler2_line = ax2.plot([], [], 'c--', lw=1.5, marker='o', markersize=4)[0]
    ruler2_text = ax2.text(0.03, 0.95, "", transform=ax2.transAxes, color='cyan', 
                           fontsize=9, bbox=dict(boxstyle="round,pad=0.3", fc="black", ec="cyan", alpha=0.8))
    ruler2_text.set_visible(False)

    # --- Widget Layout ---
    ax_left_clip     = fig.add_axes([0.05, 0.15, 0.12, 0.025])
    ax_left_low_tb   = fig.add_axes([0.18, 0.15, 0.035, 0.025])
    ax_left_high_tb  = fig.add_axes([0.22, 0.15, 0.035, 0.025])
    ax_left_gamma    = fig.add_axes([0.05, 0.10, 0.12, 0.025])
    ax_left_gamma_tb = fig.add_axes([0.18, 0.10, 0.035, 0.025])
    ax_left_reset    = fig.add_axes([0.05, 0.04, 0.05, 0.025])

    ax_left_cmap  = fig.add_axes([0.26, 0.03, 0.08, 0.15])
    ax_left_cmap.set_title("Left Cmap", fontsize=8)
    ax_mode       = fig.add_axes([0.35, 0.03, 0.09, 0.15])
    ax_mode.set_title("Mask Mode", fontsize=8)

    ax_right_clip     = fig.add_axes([0.47, 0.15, 0.12, 0.025])
    ax_right_low_tb   = fig.add_axes([0.60, 0.15, 0.035, 0.025])
    ax_right_high_tb  = fig.add_axes([0.64, 0.15, 0.035, 0.025])
    ax_right_gamma    = fig.add_axes([0.47, 0.10, 0.12, 0.025])
    ax_right_gamma_tb = fig.add_axes([0.60, 0.10, 0.035, 0.025])
    ax_right_reset    = fig.add_axes([0.47, 0.04, 0.05, 0.025])

    ax_right_cmap = fig.add_axes([0.68, 0.03, 0.08, 0.15])
    ax_right_cmap.set_title("Right Cmap", fontsize=8)

    ax_dx_tb   = fig.add_axes([0.77, 0.15, 0.035, 0.025])
    ax_dy_tb   = fig.add_axes([0.77, 0.10, 0.035, 0.025])
    ax_rin_tb  = fig.add_axes([0.86, 0.15, 0.035, 0.025])
    ax_rout_tb = fig.add_axes([0.86, 0.10, 0.035, 0.025])
    
    # Ruler Buttons
    btn_ruler1 = Button(fig.add_axes([0.15, 0.04, 0.07, 0.03]), 'Ruler Real', color='lightgray')
    btn_ruler2 = Button(fig.add_axes([0.57, 0.04, 0.07, 0.03]), 'Ruler Recip', color='lightgray')

    # High-Res Export Button
    ax_save_btn = fig.add_axes([0.80, 0.04, 0.08, 0.03])
    btn_save    = Button(ax_save_btn, f'Save #{init_idx:03d}', color='lightgreen')

    # Widgets
    s_left_clip = RangeSlider(ax_left_clip, 'Clip %', 0.0, 100.0, valinit=(0, 100))
    s_left_gamma = Slider(ax_left_gamma, 'Gamma', 0.1, 3.0, valinit=1.0)
    btn_left_reset = Button(ax_left_reset, 'Reset')

    s_right_clip = RangeSlider(ax_right_clip, 'Clip %', 0.0, 100.0, valinit=(0, 100))
    s_right_gamma = Slider(ax_right_gamma, 'Gamma', 0.1, 3.0, valinit=1.0)
    btn_right_reset = Button(ax_right_reset, 'Reset')

    radio_left_cmap = RadioButtons(ax_left_cmap, cmap_options_left, active=0)
    radio_right_cmap = RadioButtons(ax_right_cmap, cmap_options_right, active=6)
    radio_mode = RadioButtons(ax_mode, ['Sum', 'Variance', 'Fluctuation', 'COM Mag', 'COM Azimuth'], active=0)

    tb_left_low   = TextBox(ax_left_low_tb, '', initial='0.0')
    tb_left_high  = TextBox(ax_left_high_tb, '', initial='100.0')
    tb_left_gamma = TextBox(ax_left_gamma_tb, '', initial='1.0')

    tb_right_low   = TextBox(ax_right_low_tb, '', initial='0.0')
    tb_right_high  = TextBox(ax_right_high_tb, '', initial='100.0')
    tb_right_gamma = TextBox(ax_right_gamma_tb, '', initial='1.0')

    tb_dx   = TextBox(ax_dx_tb, 'dx: ', initial=str(dx))
    tb_dy   = TextBox(ax_dy_tb, 'dy: ', initial=str(dy))
    tb_rin  = TextBox(ax_rin_tb, 'r_in: ', initial=str(r_in))
    tb_rout = TextBox(ax_rout_tb, 'r_out: ', initial=str(r_out))

    updating = False

    # Initialize state dictionary
    sync_target_dict()

    # --- Callbacks ---
    def update_left(_=None):
        nonlocal updating
        processed = apply_contrast(data_left_raw, s_left_clip.val, s_left_gamma.val)
        im1.set_data(processed)
        if not updating:
            updating = True
            tb_left_low.set_val(f"{s_left_clip.val[0]:.1f}")
            tb_left_high.set_val(f"{s_left_clip.val[1]:.1f}")
            tb_left_gamma.set_val(f"{s_left_gamma.val:.2f}")
            updating = False
        fig.canvas.draw_idle()

    def update_right(_=None):
        nonlocal updating
        processed = apply_contrast(data_right_raw, s_right_clip.val, s_right_gamma.val)
        im2.set_data(processed)
        if not updating:
            updating = True
            tb_right_low.set_val(f"{s_right_clip.val[0]:.1f}")
            tb_right_high.set_val(f"{s_right_clip.val[1]:.1f}")
            tb_right_gamma.set_val(f"{s_right_gamma.val:.2f}")
            updating = False
        fig.canvas.draw_idle()

    def change_left_cmap(label):
        left_cmap['name'] = label
        im1.set_cmap(label)
        sync_target_dict()
        update_left()

    def change_right_cmap(label):
        right_cmap['name'] = label
        im2.set_cmap(label)
        sync_target_dict()
        update_right()

    def change_mode(label):
        calc_mode['mode'] = label
        if label == 'COM Azimuth':
            if left_cmap['name'] not in ['hsv', 'twilight', 'twilight_shifted']:
                try:
                    idx = cmap_options_left.index('hsv')
                    radio_left_cmap.set_active(idx)
                except ValueError:
                    pass
                im1.set_cmap('hsv')
                left_cmap['name'] = 'hsv'
                
        sync_target_dict()
        refresh_detector_mask()

    def refresh_detector_mask(_=None):
        nonlocal data_left_raw, r_in, r_out, cqx, cqy
        try:
            r_in = max(0.0, float(tb_rin.text))
            r_out = max(r_in, float(tb_rout.text))
            if r_out == r_in:
                r_out = r_in + 1.0
                tb_rout.set_val(f"{r_out:.1f}")
        except ValueError:
            return

        sync_target_dict()

        im2.set_extent(get_reciprocal_extent(cqx, cqy))
        circle_in.set_radius(r_in * rec_samp)
        circle_out.set_radius(r_out * rec_samp)

        data_left_raw, _ = compute_virtual_image(cqx, cqy, r_in, r_out, mode=calc_mode['mode'])
        ax1.set_title(f"Virtual Image [{calc_mode['mode']}]", pad=25)
        update_left()

    def refresh_spatial_roi(_=None):
        nonlocal data_right_raw, dx, dy
        try:
            dx = max(0, int(tb_dx.text))
            dy = max(0, int(tb_dy.text))
        except ValueError:
            return

        sync_target_dict()
        roi_rect.set_bounds((x - dx - 0.5) * re_samp, (y - dy - 0.5) * re_samp, 
                            (2*dx + 1) * re_samp, (2*dy + 1) * re_samp)
        data_right_raw = extract_avg_diffraction(y, x, dy, dx)
        ax2.set_title(f"Diffraction Frame ({y}, {x})", pad=25)
        update_right()

    def toggle_ruler1(_=None):
        ruler1_state['active'] = not ruler1_state['active']
        ruler1_state['p1'] = None; ruler1_state['p2'] = None; ruler1_state['length'] = 0.0
        sync_target_dict()
        if ruler1_state['active']:
            btn_ruler1.label.set_text("Real ON")
            btn_ruler1.color = 'magenta'
            ruler1_text.set_visible(True)
            ruler1_text.set_text("Click Point A on Virtual Image...")
        else:
            btn_ruler1.label.set_text("Ruler Real")
            btn_ruler1.color = 'lightgray'
            ruler1_line.set_data([], [])
            ruler1_text.set_visible(False)
        fig.canvas.draw_idle()

    def toggle_ruler2(_=None):
        ruler2_state['active'] = not ruler2_state['active']
        ruler2_state['p1'] = None; ruler2_state['p2'] = None
        ruler2_state['dq'] = 0.0; ruler2_state['d_spacing'] = 0.0
        sync_target_dict()
        if ruler2_state['active']:
            btn_ruler2.label.set_text("Recip ON")
            btn_ruler2.color = 'cyan'
            ruler2_text.set_visible(True)
            ruler2_text.set_text("Click Point A on Diffraction Pattern...")
        else:
            btn_ruler2.label.set_text("Ruler Recip")
            btn_ruler2.color = 'lightgray'
            ruler2_line.set_data([], [])
            ruler2_text.set_visible(False)
        fig.canvas.draw_idle()

    def save_figure_callback(_=None):
        """Auto-incrementing savefig export handler that preserves button click listeners."""
        curr_idx, filename = get_next_filename()
        try:
            # 1. Save high-res figure directly via Matplotlib Python engine
            fig.savefig(filename, bbox_inches='tight', pad_inches=0.05)
            print(f"Successfully exported: {filename}")
            
            # 2. Query next queued file index and update label cleanly
            next_idx, _ = get_next_filename()
            btn_save.label.set_text(f"Save #{next_idx:03d}")
            btn_save.color = 'lightblue'
        except Exception as e:
            btn_save.label.set_text("Error!")
            btn_save.color = 'salmon'
            print(f"Failed to export figure '{filename}': {e}")
            
        # 3. Force canvas & widget redraw to preserve ipympl event listener bounds
        fig.canvas.draw()
        fig.canvas.draw_idle()
        
    # --- Mouse Handlers ---
    def onclick(event):
        nonlocal cqx, cqy, x, y
        if event.inaxes == ax1:
            if ruler1_state['active']:
                pt = (event.xdata, event.ydata)
                if ruler1_state['p1'] is None or ruler1_state['p2'] is not None:
                    ruler1_state['p1'] = pt
                    ruler1_state['p2'] = None
                    ruler1_line.set_data([pt[0]], [pt[1]])
                    ruler1_text.set_text(f"P1: ({pt[0]:.2f}, {pt[1]:.2f})\nClick P2...")
                else:
                    ruler1_state['p2'] = pt
                    p1 = ruler1_state['p1']
                    ruler1_line.set_data([p1[0], pt[0]], [p1[1], pt[1]])
                    dr = np.hypot(pt[0] - p1[0], pt[1] - p1[1])
                    ruler1_state['length'] = dr
                    sync_target_dict()
                    ruler1_text.set_text(f"Δr = {dr:.3f} {re_unit}\n({dr/re_samp:.1f} px)")
                fig.canvas.draw_idle()
            else:
                click_x = int(np.clip(event.xdata / re_samp, 0, scanx - 1))
                click_y = int(np.clip(event.ydata / re_samp, 0, scany - 1))
                x, y = click_x, click_y
                v_line.set_xdata([x * re_samp, x * re_samp])
                h_line.set_ydata([y * re_samp, y * re_samp])
                refresh_spatial_roi()

        elif event.inaxes == ax2:
            if ruler2_state['active']:
                pt = (event.xdata, event.ydata)
                if ruler2_state['p1'] is None or ruler2_state['p2'] is not None:
                    ruler2_state['p1'] = pt
                    ruler2_state['p2'] = None
                    ruler2_line.set_data([pt[0]], [pt[1]])
                    ruler2_text.set_text(f"P1: ({pt[0]:.2f}, {pt[1]:.2f})\nClick P2...")
                else:
                    ruler2_state['p2'] = pt
                    p1 = ruler2_state['p1']
                    ruler2_line.set_data([p1[0], pt[0]], [p1[1], pt[1]])
                    dq = np.hypot(pt[0] - p1[0], pt[1] - p1[1])
                    d_spacing = 1.0 / dq if dq > 1e-6 else np.inf
                    ruler2_state['dq'] = dq
                    ruler2_state['d_spacing'] = d_spacing
                    sync_target_dict()
                    ruler2_text.set_text(f"Δq = {dq:.4f} {rec_unit}\nd = {d_spacing:.4f} {re_unit}")
                fig.canvas.draw_idle()
            else:
                click_qx = event.xdata / rec_samp + cqx
                click_qy = event.ydata / rec_samp + cqy
                cqx = float(np.clip(click_qx, 0, detx - 1))
                cqy = float(np.clip(click_qy, 0, dety - 1))
                refresh_detector_mask()

    def onmove(event):
        if ruler1_state['active'] and ruler1_state['p1'] is not None and ruler1_state['p2'] is None:
            if event.inaxes == ax1:
                p1 = ruler1_state['p1']
                p2 = (event.xdata, event.ydata)
                ruler1_line.set_data([p1[0], p2[0]], [p1[1], p2[1]])
                dr = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
                ruler1_state['length'] = dr
                sync_target_dict()
                ruler1_text.set_text(f"Δr = {dr:.3f} {re_unit}\n({dr/re_samp:.1f} px)")
                fig.canvas.draw_idle()

        if ruler2_state['active'] and ruler2_state['p1'] is not None and ruler2_state['p2'] is None:
            if event.inaxes == ax2:
                p1 = ruler2_state['p1']
                p2 = (event.xdata, event.ydata)
                ruler2_line.set_data([p1[0], p2[0]], [p1[1], p2[1]])
                dq = np.hypot(p2[0] - p1[0], p2[1] - p1[1])
                d_spacing = 1.0 / dq if dq > 1e-6 else np.inf
                ruler2_state['dq'] = dq
                ruler2_state['d_spacing'] = d_spacing
                sync_target_dict()
                ruler2_text.set_text(f"Δq = {dq:.4f} {rec_unit}\nd = {d_spacing:.4f} {re_unit}")
                fig.canvas.draw_idle()

    # --- Event Bindings ---
    s_left_clip.on_changed(update_left); s_left_gamma.on_changed(update_left)
    s_right_clip.on_changed(update_right); s_right_gamma.on_changed(update_right)
    btn_left_reset.on_clicked(lambda _: (s_left_clip.set_val((0, 100)), s_left_gamma.set_val(1.0)))
    btn_right_reset.on_clicked(lambda _: (s_right_clip.set_val((0, 100)), s_right_gamma.set_val(1.0)))
    
    radio_left_cmap.on_clicked(change_left_cmap)
    radio_right_cmap.on_clicked(change_right_cmap)
    radio_mode.on_clicked(change_mode)
    
    btn_ruler1.on_clicked(toggle_ruler1)
    btn_ruler2.on_clicked(toggle_ruler2)

    btn_save.on_clicked(save_figure_callback)
    
    tb_dx.on_submit(refresh_spatial_roi); tb_dy.on_submit(refresh_spatial_roi)
    tb_rin.on_submit(refresh_detector_mask); tb_rout.on_submit(refresh_detector_mask)

    fig.canvas.mpl_connect('button_press_event', onclick)
    fig.canvas.mpl_connect('motion_notify_event', onmove)

    fig._widgets = [
        s_left_clip, s_left_gamma, s_right_clip, s_right_gamma,
        tb_left_low, tb_left_high, tb_left_gamma,
        tb_right_low, tb_right_high, tb_right_gamma,
        tb_dx, tb_dy, tb_rin, tb_rout, btn_left_reset, btn_right_reset, 
        radio_left_cmap, radio_right_cmap, radio_mode, btn_ruler1, btn_ruler2, btn_save
    ]

    plt.ion()
    plt.show()
    return target_dict

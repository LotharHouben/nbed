import json
import os
import re


# --- Calibration Manager with Binning Correction ---
class MicroscopeCalibrationManager:
    def __init__(self, db_path):
        self.db_path = db_path
        self.db = self._load_db()
        self.current_table = None

    def _load_db(self):
        try:
            with open(self.db_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading calibration database '{self.db_path}': {e}")
            return {}

    def set_current_table(self, instrument, device, voltage, mode="STEM", submode="Microprobe", imaging_type="Magnification"):
        self.current_table = None
        tables = self.db.get("instrumenttables", {})
        inst_dict = tables.get(instrument, {})
        dev_list = inst_dict.get(device, [])
        volt_str = str(voltage)

        for entry in dev_list:
            if entry.get("mode") == mode and entry.get("submode") == submode:
                voltages = entry.get("voltage", {})
                if volt_str in voltages:
                    target_volt = voltages[volt_str]
                    if imaging_type in target_volt:
                        self.current_table = target_volt[imaging_type]
                        return True
        return False

    def evaluate_interpolation(self, formula_str, x_val):
        try:
            if ':' in formula_str:
                expr = formula_str.split(':', 1)[1].strip()
            else:
                expr = formula_str.strip()
            expr_eval = re.sub(r'(?<![a-zA-Z0-9_])x(?![a-zA-Z0-9_])', str(float(x_val)), expr)
            return float(eval(expr_eval, {"__builtins__": None}, {}))
        except Exception as e:
            print(f"Failed to evaluate formula '{formula_str}' with x={x_val}: {e}")
            return None

    def get_scale_from_current_table(self, nominal_val):
        if not self.current_table:
            return None, "No Table Selected"

        values_dict = self.current_table.get("values", {})
        nom_str = str(nominal_val)

        if nom_str in values_dict:
            return float(values_dict[nom_str]), "Exact Match"

        for k, val in values_dict.items():
            try:
                if abs(float(k) - float(nominal_val)) < 1e-4:
                    return float(val), f"Exact Match ({k})"
            except ValueError:
                continue

        formula = self.current_table.get("interpolation")
        if formula:
            fov = self.evaluate_interpolation(formula, nominal_val)
            if fov is not None:
                return fov, "Interpolation Formula"

        return None, "Lookup Failed"

    def get_calibration_from_metadata(self, md, type='DECTRIS', bin_scan=(1, 1), bin_det=(1, 1)):
        """
        Obtains FOV sampling directly from DECTRIS metadata and corrects for binning factors:
        bin_scan: tuple (bin_scan_y, bin_scan_x)
        bin_det: tuple (bin_det_y, bin_det_x)
        """
        def to_str(val):
            if isinstance(val, bytes):
                return val.decode('utf-8')
            if hasattr(val, 'item'):
                val = val.item()
                if isinstance(val, bytes):
                    return val.decode('utf-8')
            return str(val)

        samp, qsamp = None, None

        if type == 'DECTRIS':
            instrument = to_str(md['electron_microscope']['model'])
            device = to_str(md['entry']['instrument']['detector']['description'])
            cl = float(md['electron_microscope']['imaging_system']['camera_length']) * 1000.0  # m to mm
            mag = float(md['electron_microscope']['illumination_system']['scan_magnification'])
            mode = to_str(md['electron_microscope']['illumination_system']['mode'])
            voltage = int(md['electron_microscope']['electron_source']['accelerating_voltage'])

            # Extract binning multipliers (using X axis binning factor)
            bin_scan_x = float(bin_scan[1]) if isinstance(bin_scan, (tuple, list)) else float(bin_scan)
            bin_det_x = float(bin_det[1]) if isinstance(bin_det, (tuple, list)) else float(bin_det)

            # 1. Real-Space Sampling Calculation
            if self.set_current_table(instrument, device, voltage, mode, submode="Microprobe", imaging_type="Magnification"):
                scale, source = self.get_scale_from_current_table(mag)
                if scale:
                    # Unbinned pixels in original scan grid
                    unbinned_nx = float(md['electron_microscope']['scan_controller']['regular_scan']['n_pixels_x'])
                    raw_samp = scale / unbinned_nx
                    # Correct for binning factor
                    samp = raw_samp * bin_scan_x
                    print(f"Real-Space Calibration: FOV = {scale:.2f} nm, Raw = {raw_samp:.4f} nm/px, Binned ({bin_scan_x}x) = {samp:.4f} nm/px ({source})")

            # 2. Reciprocal-Space Sampling Calculation
            if self.set_current_table(instrument, device, voltage, mode, submode="Microprobe", imaging_type="Cameralength"):
                qscale, source = self.get_scale_from_current_table(cl)
                if qscale:
                    # Unbinned detector pixels
                    unbinned_det_x = float(md['entry']['instrument']['detector']['module']['data_size'][0])
                    raw_qsamp = qscale / unbinned_det_x
                    # Correct for binning factor
                    qsamp = raw_qsamp * bin_det_x
                    print(f"Reciprocal-Space Calibration: FOV = {qscale:.2f} 1/nm, Raw = {raw_qsamp:.4f} (1/nm)/px, Binned ({bin_det_x}x) = {qsamp:.4f} (1/nm)/px ({source})")

        return samp, qsamp

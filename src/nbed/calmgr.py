import json
import os
import re


class MicroscopeCalibrationManager:
    """
    Manages microscope calibration tables loaded from a JSON file string.
    Supports direct lookup matching against discrete values before falling back to interpolation.
    """
    def __init__(self, file_path: str):
        self.file_path = file_path
        self.current_table = None
        self._calibrations = self._load_file(file_path)

    def _load_file(self, file_path: str) -> dict:
        """Loads and parses the JSON calibration file with error handling."""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Calibration file not found at path: '{file_path}'")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if not isinstance(data, dict):
                    raise ValueError("JSON root element must be an object/dictionary.")
                return data
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse JSON calibration file '{file_path}': {e}") from e
        except Exception as e:
            raise IOError(f"Unexpected error reading file '{file_path}': {e}") from e

    @property
    def calibration_data(self) -> dict:
        return self._calibrations

    def get_calibration(
        self, 
        instrument: str, 
        device: str, 
        voltage: int | str, 
        mode: str, 
        imaging_type: str, 
        submode: str = None, 
        subsetting: str = None
    ) -> dict:
        """Retrieves the calibration record matching the specified microscope criteria."""
        if not self._calibrations:
            raise RuntimeError("No calibration data is currently loaded.")

        tables = self._calibrations.get("instrumenttables")
        if not isinstance(tables, dict):
            raise KeyError("Invalid calibration format: 'instrumenttables' key missing or invalid.")

        if instrument not in tables:
            available_instruments = list(tables.keys())
            raise KeyError(
                f"Instrument '{instrument}' not found. Available instruments: {available_instruments}"
            )

        device_entries = tables[instrument].get(device)
        if not device_entries:
            available_devices = list(tables[instrument].keys())
            raise KeyError(
                f"Device '{device}' not found for '{instrument}'. Available devices: {available_devices}"
            )

        if not isinstance(device_entries, list):
            raise TypeError(f"Expected list of device configurations for '{device}', got {type(device_entries).__name__}.")

        str_voltage = str(voltage)

        matching_entry = None
        for entry in device_entries:
            if not isinstance(entry, dict):
                continue
            if entry.get("mode") != mode:
                continue
            if submode is not None and entry.get("submode") != submode:
                continue
            if subsetting is not None and entry.get("subsetting") != subsetting:
                continue
            matching_entry = entry
            break

        if not matching_entry:
            raise KeyError(
                f"No matching entry found for Mode='{mode}', Submode='{submode}', "
                f"Subsetting='{subsetting}' under device '{device}'."
            )

        voltages_dict = matching_entry.get("voltage")
        if not isinstance(voltages_dict, dict):
            raise KeyError("No valid 'voltage' dictionary found in the matching configuration.")

        voltage_record = voltages_dict.get(str_voltage)
        if not voltage_record:
            available_voltages = list(voltages_dict.keys())
            raise KeyError(
                f"Voltage '{str_voltage}' not found. Available voltages for this mode: {available_voltages}"
            )

        if imaging_type not in voltage_record:
            available_types = list(voltage_record.keys())
            raise KeyError(
                f"Imaging type '{imaging_type}' not found. Available types for {str_voltage}V: {available_types}"
            )

        return voltage_record[imaging_type]

    def set_current_table(
        self, 
        instrument: str, 
        device: str, 
        voltage: int | str, 
        mode: str, 
        imaging_type: str, 
        submode: str = None, 
        subsetting: str = None
    ) -> dict:
        """Retrieves calibration dict and locks it in as 'current_table'."""
        try:
            self.current_table = self.get_calibration(
                instrument=instrument,
                device=device,
                voltage=voltage,
                mode=mode,
                imaging_type=imaging_type,
                submode=submode,
                subsetting=subsetting
            )
            return self.current_table
        except (KeyError, TypeError, ValueError, RuntimeError) as e:
            self.current_table = None
            raise ValueError(f"Failed to set current calibration table: {e}") from e

    def get_interpolation_from_current_table(self, nominal_value: float | int) -> float:
        """Evaluates the interpolation formula stored in 'current_table' for input x."""
        if self.current_table is None:
            raise RuntimeError("No active calibration set. Call 'set_current_table()' first.")

        formula_str = self.current_table.get("interpolation")
        if not formula_str or not isinstance(formula_str, str):
            raise KeyError("The active calibration table is missing a valid 'interpolation' string.")

        try:
            x_val = float(nominal_value)
        except (ValueError, TypeError) as e:
            raise TypeError(f"Nominal value '{nominal_value}' must be a valid number.") from e

        if ":" in formula_str:
            formula = formula_str.split(":", 1)[1].strip()
        else:
            formula = formula_str.strip()

        sanitized = re.sub(r'\s+', '', formula)
        if not re.match(r'^[0-9x\+\-\*\/\.\(\)E e]+$', sanitized, re.IGNORECASE):
            raise ValueError(f"Unsafe or malformed mathematical formula string: '{formula_str}'")

        try:
            result = eval(formula, {"__builtins__": None}, {"x": x_val})
            return float(result)
        except ZeroDivisionError as e:
            raise ZeroDivisionError(f"Division by zero in interpolation formula '{formula}' for x={x_val}") from e
        except Exception as e:
            raise ValueError(f"Failed to evaluate formula '{formula}' with x={x_val}: {e}") from e

    def get_scale_from_current_table(self, nominal_value: float | int, rtol: float = 1e-4) -> tuple[float, str]:
        """
        Attempts to match nominal_value against discrete values in 'current_table'.
        If found (within relative tolerance rtol), returns the exact calibrated value.
        If not found, reverts to evaluating the interpolation formula.

        Returns:
            tuple: (calibrated_scale, match_source) where match_source is 'exact_match' or 'interpolated'
        """
        samp=1.
        qsamp=1.
        if self.current_table is None:
            raise RuntimeError("No active calibration set. Call 'set_current_table()' first.")

        try:
            x_val = float(nominal_value)
        except (ValueError, TypeError) as e:
            raise TypeError(f"Nominal value '{nominal_value}' must be a valid number.") from e

        # 1. Check discrete "values" dictionary for an exact or near match
        discrete_values = self.current_table.get("values", {})
        if isinstance(discrete_values, dict):
            for key, cal_val in discrete_values.items():
                try:
                    key_val = float(key)
                    # Check relative tolerance (handles string keys like "1146.3" or "520.")
                    if abs(key_val - x_val) <= rtol * max(abs(key_val), 1e-9):
                        return float(cal_val), "exact_match"
                except (ValueError, TypeError):
                    continue

        # 2. Fall back to evaluation of interpolation formula
        interpolated_val = self.get_interpolation_from_current_table(x_val)
        return interpolated_val, "interpolated"
    
    def get_calibration_from_metadata(self, md, type='DECTRIS'):
        """Try to obtain calibration data from metadata dictionary."""

        def to_str(val):
            """Converts bytes, byte arrays, or NumPy scalars to standard Python str."""
            if isinstance(val, bytes):
                return val.decode('utf-8')
            if hasattr(val, 'item'):  # Handles 0D NumPy arrays / scalars
                val = val.item()
                if isinstance(val, bytes):
                    return val.decode('utf-8')
            return str(val)

        samp, qsamp = None, None

        if type == 'DECTRIS':
            instrument = to_str(md['electron_microscope']['model'])
            device = to_str(md['entry']['instrument']['detector']['description'])
            cl = md['electron_microscope']['imaging_system']['camera_length'] * 1000
            mag = md['electron_microscope']['illumination_system']['scan_magnification']
            mode = to_str(md['electron_microscope']['illumination_system']['mode'])  # Removed extra trailing ')'
            voltage = int(md['electron_microscope']['electron_source']['accelerating_voltage'])

            # 1. Magnification Calibration
            self.set_current_table(
                instrument=instrument,
                device=device,
                voltage=voltage,
                mode=mode,
                submode="Microprobe",
                imaging_type="Magnification"
            )

            scale, source = self.get_scale_from_current_table(mag)
            samp = scale / float(md['electron_microscope']['scan_controller']['regular_scan']['n_pixels_x'])
            print(f"Magnification calibration Value FOV/sampling: {mag} -> {scale} / {samp} ({source})")

            # 2. Camera Length / Diffraction Calibration
            self.set_current_table(
                instrument=instrument,
                device=device,
                voltage=voltage,
                mode=mode,
                submode="Microprobe",
                imaging_type="Cameralength"
            )

            qscale, source = self.get_scale_from_current_table(cl)
            qsamp = qscale / float(md['entry']['instrument']['detector']['module']['data_size'][0])
            print(f"Diffraction calibration Value FOV/sampling: {cl} -> {qscale} / {qsamp} ({source})")

        return samp, qsamp
